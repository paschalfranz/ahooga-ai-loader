import os
import time
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

load_dotenv()

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX")

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

DEBUG_RETRIEVAL = False

# Optional frontend image URLs
BAFANG_MIVICE_IMAGE_URL = os.getenv("BAFANG_MIVICE_IMAGE_URL", "").strip()
BAFANG_DISPLAY_IMAGE_URL = os.getenv("BAFANG_DISPLAY_IMAGE_URL", "").strip()
MIVICE_DISPLAY_IMAGE_URL = os.getenv("MIVICE_DISPLAY_IMAGE_URL", "").strip()

# In-memory conversation state
# conversation_id -> state
CONVERSATION_STATE: Dict[str, Dict[str, object]] = {}
STATE_TTL_SECONDS = 60 * 30


# =========================================================
# ENV / CLIENTS
# =========================================================

def validate_env():
    required = {
        "AZURE_SEARCH_ENDPOINT": SEARCH_ENDPOINT,
        "AZURE_SEARCH_ADMIN_KEY": SEARCH_KEY,
        "AZURE_SEARCH_INDEX": INDEX_NAME,
        "AZURE_OPENAI_ENDPOINT": OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": OPENAI_KEY,
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,
        "AZURE_OPENAI_CHAT_DEPLOYMENT": CHAT_DEPLOYMENT,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")


validate_env()

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_KEY),
)

openai_client = AzureOpenAI(
    api_key=OPENAI_KEY,
    api_version="2024-02-01",
    azure_endpoint=OPENAI_ENDPOINT,
)


# =========================================================
# CONSTANTS
# =========================================================

SAFE_RETURN_FALLBACK = (
    "I do not have a reliable return procedure in the current knowledge base. "
    "The return process usually depends on where the bike was purchased."
)

SAFE_POWER_FALLBACK_ON = (
    "Start with the basic checks: make sure the battery has charge, unfold the bike fully, "
    "hold the display power button for a few seconds, and inspect the visible cables and connectors "
    "around the handlebar and frame entry points. If nothing changes after that, the issue likely "
    "needs a more specific diagnosis."
)

SAFE_POWER_FALLBACK_OFF = (
    "Start with the basic checks: confirm the battery is charged, then inspect the visible cables and connectors, "
    "especially around the folding areas and handlebar area. If the bike switches off again after those checks, "
    "the next step depends on which electrical system the bike has."
)

BOTH_COMPONENTS = [
    "rack",
    "rear light",
    "taillight",
    "tail light",
    "rear hinge axle",
    "front hinge axis",
    "front handle clamp",
    "twist and lock",
    "twist-and-lock",
    "stem",
    "battery lock",
    "battery-lock",
    "saddle sleeve",
    "saddle-sleeve",
    "ffi",
    "front clasp",
    "clasp",
]

DEFINITION_COMPONENTS = {
    "controller",
    "display",
    "hmi",
    "torque-sensor",
    "tool",
    "app",
    "battery",
    "powertrain",
}


# =========================================================
# STATE
# =========================================================

def cleanup_state() -> None:
    now = time.time()
    expired = [
        key for key, value in CONVERSATION_STATE.items()
        if now - float(value.get("created_at", 0)) > STATE_TTL_SECONDS
    ]
    for key in expired:
        CONVERSATION_STATE.pop(key, None)


def ensure_state(conversation_id: Optional[str]) -> Optional[Dict[str, object]]:
    if not conversation_id:
        return None
    cleanup_state()
    state = CONVERSATION_STATE.get(conversation_id)
    if not state:
        state = {
            "pending_question": "",
            "last_user_question": "",
            "last_answer": "",
            "last_system": "",
            "last_component": "",
            "created_at": time.time(),
        }
        CONVERSATION_STATE[conversation_id] = state
    else:
        state["created_at"] = time.time()
    return state


def remember_pending_question(conversation_id: Optional[str], question: str) -> None:
    state = ensure_state(conversation_id)
    if not state:
        return
    state["pending_question"] = question
    state["created_at"] = time.time()


def get_pending_question(conversation_id: Optional[str]) -> Optional[str]:
    state = ensure_state(conversation_id)
    if not state:
        return None
    value = str(state.get("pending_question") or "").strip()
    return value or None


def clear_pending_question(conversation_id: Optional[str]) -> None:
    state = ensure_state(conversation_id)
    if not state:
        return
    state["pending_question"] = ""


def remember_turn(
    conversation_id: Optional[str],
    user_question: str,
    answer: str,
    forced_system: Optional[str] = None,
    component: Optional[str] = None,
) -> None:
    state = ensure_state(conversation_id)
    if not state:
        return
    state["last_user_question"] = user_question
    state["last_answer"] = answer
    state["last_system"] = forced_system or ""
    state["last_component"] = component or ""
    state["created_at"] = time.time()


def get_last_user_question(conversation_id: Optional[str]) -> Optional[str]:
    state = ensure_state(conversation_id)
    if not state:
        return None
    value = str(state.get("last_user_question") or "").strip()
    return value or None


def get_last_answer(conversation_id: Optional[str]) -> Optional[str]:
    state = ensure_state(conversation_id)
    if not state:
        return None
    value = str(state.get("last_answer") or "").strip()
    return value or None


def get_last_system(conversation_id: Optional[str]) -> Optional[str]:
    state = ensure_state(conversation_id)
    if not state:
        return None
    value = str(state.get("last_system") or "").strip()
    return value or None


def get_last_component(conversation_id: Optional[str]) -> Optional[str]:
    state = ensure_state(conversation_id)
    if not state:
        return None
    value = str(state.get("last_component") or "").strip()
    return value or None


# =========================================================
# EMBEDDING
# =========================================================

def embed_text(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding


# =========================================================
# NORMALIZATION / QUESTION ANALYSIS
# =========================================================

def normalize_question(text: str) -> str:
    q = text.lower().strip()

    replacements = {
        "mivece": "mivice",
        "bafeng": "bafang",
        "what’s": "what's",
        "max’s": "max",
        "older max": "old max",
        "newer max": "new max",
        "doesnt": "doesn't",
        "wont": "won't",
        "ontroller": "controller",
        "old oe new": "old or new",
        "tecnical": "technical",
        "undersatnding": "understanding",
        "e system": "electric system",
        "e-system": "electric system",
    }

    for old, new in replacements.items():
        q = q.replace(old, new)

    return " ".join(q.split())


def detect_system_from_text(text: str) -> Optional[str]:
    q = normalize_question(text)

    if "besst" in q:
        return "Bafang"

    if any(x in q for x in ["both", "either", "all max", "all versions"]):
        return "Both"

    if any(x in q for x in [
        "bafang",
        "old max",
        "nexus",
        "old system",
        "older system",
        "old supplier",
    ]):
        return "Bafang"

    if any(x in q for x in [
        "mivice",
        "new max",
        "alfine",
        "derailleur",
        "new system",
        "newer system",
        "new supplier",
    ]):
        return "Mivice"

    if q in {"old", "older"}:
        return "Bafang"

    if q in {"new", "newer"}:
        return "Mivice"

    return None


def detect_doc_type_from_text(text: str) -> Optional[str]:
    q = normalize_question(text)

    if "warranty" in q or "guarantee" in q or "claim" in q:
        return "warranty"

    if any(x in q for x in [
        "app", "display", "firmware", "software update",
        "besst", "tool", "bluetooth", "pair", "screen", "hmi"
    ]):
        return "system"

    if any(x in q for x in ["part number", "part numbers", "parts list", "spare part", "parts"]):
        return "parts"

    if "manual" in q or "owner" in q:
        return "manual"

    if any(x in q for x in [
        "install", "assembly", "assemble", "replace", "remove",
        "reinstall", "change", "fit", "mount"
    ]):
        return "assembly"

    if any(x in q for x in [
        "error", "issue", "problem", "troubleshoot", "fault", "blinking",
        "turn on", "turn off", "does not turn on", "turns off",
        "electric system", "electric support", "no support", "no assist"
    ]):
        return "troubleshooting"

    return None


def detect_component_from_text(text: str) -> Optional[str]:
    q = normalize_question(text)

    if "warranty" in q or "guarantee" in q or "claim" in q:
        return "warranty"
    if "controller" in q:
        return "controller"
    if "torque sensor" in q or "torque" in q:
        return "torque-sensor"
    if "front clasp" in q or ("clasp" in q and "front" in q):
        return "front-clasp"
    if "ffi" in q:
        return "ffi"
    if "bms" in q:
        return "bms"
    if "battery lock" in q or "battery-lock" in q:
        return "battery-lock"
    if "battery" in q:
        return "battery"
    if "rear light" in q or "taillight" in q or "tail light" in q or "light is blinking" in q:
        return "rear-light"
    if "rack cable extension" in q:
        return "rack-cable-extension"
    if "rack" in q:
        return "rack"
    if "rear hinge axle" in q:
        return "rear-hinge-axle"
    if "front hinge axis" in q:
        return "front-hinge-axis"
    if "front handle clamp" in q:
        return "front-handle-clamp"
    if "twist and lock" in q or "twist-and-lock" in q:
        return "twist-and-lock"
    if "stem" in q:
        return "stem"
    if "saddle sleeve" in q or "saddle-sleeve" in q:
        return "saddle-sleeve"
    if "front wheel" in q:
        return "front-wheel"
    if "cable management" in q or "manage the cables" in q or "cables" in q:
        return "cable-management"
    if "derailleur hanger" in q:
        return "foldable-derailleur-hanger"
    if "display" in q or "screen" in q or "hmi" in q:
        return "display"
    if "app" in q or "bluetooth" in q or "pair" in q:
        return "app"
    if "firmware" in q or "software update" in q or "update" in q:
        return "update"
    if "besst" in q or "tool" in q:
        return "tool"
    if "part number" in q or "parts list" in q or "parts" in q:
        return "general"
    if any(x in q for x in ["electric system", "electric support", "no support", "no assist", "motor not helping"]):
        return "powertrain"
    if "error code" in q or "error codes" in q or "error" in q or "blinking" in q:
        return "powertrain"
    if "motor" in q or "powertrain" in q:
        return "powertrain"

    return None


def detect_user_type(text: str) -> str:
    q = normalize_question(text)

    dealer_signals = [
        "part number", "parts list", "controller replacement", "torque sensor replacement",
        "dealer", "retailer", "warranty portal", "diagnostic", "besst", "connector",
        "serial number", "service bulletin", "credit note", "replacement part",
    ]
    if any(x in q for x in dealer_signals):
        return "dealer"

    return "customer"


def is_return_question(text: str) -> bool:
    q = normalize_question(text)
    return any(x in q for x in [
        "return my bike", "return bike", "return the bike",
        "refund", "send back", "cancel order", "give back my bike",
        "can i return my bike"
    ])


def is_warranty_question(text: str) -> bool:
    q = normalize_question(text)
    return any(x in q for x in [
        "warranty", "guarantee", "claim", "ticket", "defect", "defective part"
    ])


def is_technical_question(text: str) -> bool:
    q = normalize_question(text)

    if is_definition_question(q):
        return False

    technical_terms = [
        "controller", "torque", "sensor", "display", "firmware",
        "besst", "connector", "cable", "replace", "install",
        "remove", "assembly", "motor", "powertrain", "hanger",
        "rear light", "rack", "battery", "error", "diagnostic"
    ]
    return any(term in q for term in technical_terms)


def is_vague_power_issue(text: str) -> bool:
    q = normalize_question(text)
    return any(x in q for x in [
        "does not turn on",
        "doesn't turn on",
        "won't turn on",
        "will not turn on",
        "bike turns off",
        "turns off",
        "turn off",
        "bike switched off",
        "bike shuts off",
        "turns off randomly"
    ])


def is_definition_question(text: str) -> bool:
    q = normalize_question(text)
    return (
        q.startswith("what is ")
        or q.startswith("what's ")
        or q.startswith("what does ")
        or q.startswith("what do you mean by ")
        or q.startswith("explain ")
        or q.startswith("describe ")
    )


def is_identification_question(text: str) -> bool:
    q = normalize_question(text)
    patterns = [
        "how do i know",
        "how can i know",
        "how do i identify",
        "which system do i have",
        "is it old or new",
        "is it bafang or mivice",
        "how to tell if",
        "what system do i have",
        "difference between old and new",
        "difference between the older and newer",
        "difference between bafang and mivice",
        "how do i identify if my max has the old or new system",
        "what's the difference between the older and newer max systems",
        "what is the difference between bafang and mivice",
    ]
    return any(p in q for p in patterns)


def is_display_blinking_question(text: str) -> bool:
    q = normalize_question(text)
    return "display" in q and "blinking" in q


def is_repair_or_install_question(text: str) -> bool:
    q = normalize_question(text)
    return any(x in q for x in [
        "how do i change",
        "how do i replace",
        "how do i install",
        "replace",
        "installation",
        "install a new",
        "remove",
        "reinstall",
        "mount",
        "fit",
    ])


def is_location_question(text: str) -> bool:
    q = normalize_question(text)
    return any(x in q for x in [
        "where is",
        "where can i find",
        "where do i find",
        "where can i locate",
    ])


def is_standalone_system_reply(text: str) -> bool:
    q = normalize_question(text)
    return q in {"bafang", "mivice", "both", "old", "new", "older", "newer"}


def is_rewrite_followup(text: str) -> bool:
    q = normalize_question(text)
    patterns = [
        "make it less technical",
        "less technical",
        "simplify",
        "make it simpler",
        "explain it simply",
        "customer version",
        "make it customer friendly",
        "retailer version",
        "dealer version",
    ]
    return any(p in q for p in patterns)


def is_pronoun_followup(text: str) -> bool:
    q = normalize_question(text)
    patterns = [
        "how do you use it",
        "how do i use it",
        "where is it",
        "how do i fix it",
        "how do i replace it",
        "how do i install it",
        "how does it work",
        "what does that mean",
    ]
    return any(p in q for p in patterns)


def is_contextual_followup(text: str) -> bool:
    q = normalize_question(text)
    patterns = [
        "but how will a customer know",
        "what happens then",
        "what happens after that",
        "and then",
        "what about for a customer",
        "what about for the retailer",
        "what about for a dealer",
        "do customers need to know",
    ]
    return any(p in q for p in patterns)


def is_electrical_symptom_question(text: str) -> bool:
    q = normalize_question(text)
    patterns = [
        "electric system",
        "electric support",
        "electric assist",
        "e support",
        "e-system",
        "no support",
        "no assist",
        "motor not helping",
        "assist not working",
        "support not working",
        "my bike light is blinking",
        "light is blinking",
    ]
    return any(p in q for p in patterns)


def needs_system_clarification(question: str, forced_system: Optional[str]) -> bool:
    if forced_system:
        return False

    q = normalize_question(question)

    if "besst" in q:
        return False

    if is_definition_question(q) or is_identification_question(q):
        return False

    if is_warranty_question(q):
        return False

    if any(x in q for x in BOTH_COMPONENTS):
        return False

    if any(x in q for x in ["does not work", "not working", "issue", "problem"]):
        return False

    system_dependent_topics = [
        "controller", "torque", "sensor", "display", "app", "pair", "bluetooth",
        "firmware", "update", "powertrain", "motor",
        "error", "blinking", "turn on", "turn off"
    ]
    return any(x in q for x in system_dependent_topics)


# =========================================================
# FRONTEND IMAGE HELPERS
# =========================================================

def maybe_add_image_note(base_text: str, image_url: str) -> str:
    if image_url:
        return f"{base_text}\n\nReference image: {image_url}"
    return base_text


# =========================================================
# SIMPLE DEFINITIONS
# =========================================================

def is_simple_definition(question: str) -> bool:
    q = normalize_question(question)

    if not is_definition_question(q):
        return False

    keywords = [
        "controller", "hmi", "display", "torque sensor", "battery", "motor", "besst"
    ]
    return any(k in q for k in keywords)


def definition_facts(question: str) -> Optional[str]:
    q = normalize_question(question)

    if "controller" in q:
        return "A controller is the electronic unit that manages how the battery, motor and display work together on the bike."
    if "hmi" in q:
        return "HMI means Human-Machine Interface. On the bike, that usually means the display or control unit the rider uses to switch the bike on, view information and change assistance settings."
    if "besst" in q and ("use" in q or "how" in q):
        return "BESST is mainly a technician tool for older Bafang-based systems. It is used to connect to the bike, read system information, check faults and carry out service actions."
    if "besst" in q:
        return "BESST is a Bafang diagnostic and service tool used mainly by technicians on older Bafang-based systems."
    if "torque sensor" in q and ("how does" in q or "how it works" in q or "how does it work" in q):
        return "A torque sensor measures how much force the rider puts into the pedals. The system uses that signal to decide how much motor support to give."
    if "torque sensor" in q or "torque" in q:
        return "A torque sensor measures how hard the rider is pedaling so the bike can adjust motor support."
    if "display" in q:
        return "The display is the screen or control unit on the bike that shows information such as battery level, speed and assistance settings."
    if "battery" in q:
        return "The battery stores the electrical energy used to power the bike’s assist system."
    if "motor" in q:
        return "The motor is the part of the e-bike system that provides electrical assist when you pedal."
    return None


def generate_definition_answer(question: str) -> Optional[str]:
    facts = definition_facts(question)
    if not facts:
        return None

    response = openai_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Ahooga after-sales assistant. "
                    "Write a useful, customer-friendly explanation using only the provided facts. "
                    "Do not use marketing language. "
                    "Do not say things like handy, seamlessly, smooth experience, efficient, or feel free to ask. "
                    "Keep it natural, practical, and slightly varied in wording. "
                    "Usually answer in 2 to 4 sentences."
                )
            },
            {
                "role": "user",
                "content": f"Facts: {facts}"
            }
        ],
        temperature=0.45,
    )

    return response.choices[0].message.content.strip()


# =========================================================
# SPECIAL STATIC ANSWERS
# =========================================================

def answer_identification_clarifier() -> str:
    answer = (
        "Is your MAX the older Bafang system or the newer Mivice system? "
        "The display is usually the quickest visible clue."
    )
    return maybe_add_image_note(answer, BAFANG_MIVICE_IMAGE_URL)


def answer_vague_power_issue(question: str, forced_system: Optional[str]) -> str:
    q = normalize_question(question)

    if "turn off" in q or "turns off" in q or "shuts off" in q:
        if forced_system:
            return (
                f"The bike switching off points to a system-specific electrical issue. "
                f"Start with the battery charge and the visible cable and connector checks. "
                f"Then continue with the troubleshooting steps for the {forced_system} system."
            )
        return SAFE_POWER_FALLBACK_OFF

    if forced_system:
        return (
            f"Start with the battery charge, full unfolding of the bike, and the visible cable and connector checks. "
            f"After that, the next steps should follow the {forced_system} system guidance."
        )

    return SAFE_POWER_FALLBACK_ON


def answer_return_question() -> str:
    return SAFE_RETURN_FALLBACK


def answer_controller_location_fallback(forced_system: Optional[str]) -> str:
    if forced_system == "Bafang":
        return (
            "On the older MAX with the Bafang system, the controller is housed in the front headtube area, "
            "behind the front assembly where the electrical connections are routed."
        )

    if forced_system == "Mivice":
        return (
            "On the newer MAX with the Mivice system, the controller is also housed in the front headtube area, "
            "where the front electrical routing and connectors are managed."
        )

    return (
        "On MAX bikes, the controller is generally housed in the front headtube area. "
        "The exact setup differs between the older Bafang system and the newer Mivice system."
    )


def answer_compare_systems() -> str:
    answer = (
        "On MAX bikes, Bafang refers to the older electrical system and Mivice refers to the newer one. "
        "In practice, that means they use different displays, different electronic components, and different service procedures. "
        "So before troubleshooting or replacing parts, it is important to know which system the bike has."
    )
    return maybe_add_image_note(answer, BAFANG_MIVICE_IMAGE_URL)


def answer_identify_old_or_new() -> str:
    answer = (
        "The quickest way is to look at the display. "
        "The older MAX uses the Bafang system and the newer MAX uses the Mivice system, "
        "so the display is usually the easiest visible clue."
    )
    return maybe_add_image_note(answer, BAFANG_MIVICE_IMAGE_URL)


# =========================================================
# FOLLOW-UP HELPERS
# =========================================================

def rewrite_previous_answer(user_input: str, previous_answer: str) -> str:
    q = normalize_question(user_input)

    if "retailer" in q or "dealer" in q:
        audience = "retailer"
    elif "customer" in q:
        audience = "customer"
    else:
        audience = "customer"

    if "less technical" in q or "simpl" in q or "customer" in q:
        instruction = "Rewrite the answer in simpler, less technical language."
    else:
        instruction = "Rewrite the answer clearly while keeping the meaning."

    response = openai_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Ahooga after-sales assistant. "
                    "Rewrite the previous answer only. "
                    "Do not introduce new facts. "
                    "Keep the meaning the same. "
                    "Make it suitable for the requested audience."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Audience: {audience}\n"
                    f"Instruction: {instruction}\n\n"
                    f"Previous answer:\n{previous_answer}"
                )
            }
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def expand_followup_question(user_input: str, conversation_id: Optional[str]) -> Optional[str]:
    q = normalize_question(user_input)
    last_question = get_last_user_question(conversation_id)
    last_component = get_last_component(conversation_id)
    last_system = get_last_system(conversation_id)

    if not last_question:
        return None

    system_suffix = f" ({last_system})" if last_system else ""

    if is_pronoun_followup(q):
        if "use it" in q:
            if last_component == "tool":
                return f"How do I use the BESST tool{system_suffix}?"
            if last_component:
                return f"How do I use the {last_component}{system_suffix}?"
            return last_question

        if "where is it" in q:
            if last_component:
                return f"Where is the {last_component} on my MAX{system_suffix}?"
            return last_question

        if "how does it work" in q:
            if last_component:
                return f"How does the {last_component} work{system_suffix}?"
            return last_question

    if is_contextual_followup(q):
        if "customer know" in q:
            return f"For a customer, {last_question}"
        if "retailer" in q or "dealer" in q:
            return f"For a retailer, {last_question}"
        return last_question

    return None


# =========================================================
# QUERY EXPANSION / FILTERING
# =========================================================

def expand_query(question: str) -> str:
    q = normalize_question(question)

    if is_warranty_question(q):
        q += " warranty portal claim defective component frame number invoice pictures process coverage credit note retailer"

    if "display" in q or "hmi" in q:
        q += " display screen hmi d101 dpe180 e181 usage settings buttons"

    if "app" in q or "pair" in q or "bluetooth" in q:
        q += " app pairing bluetooth goplus instructions"

    if "firmware" in q or "update" in q or "software" in q:
        q += " firmware software update besst instructions steps"

    if "manual" in q:
        q += " guide instructions documentation owner manual"

    if "part number" in q or "parts list" in q or "parts" in q:
        q += " part numbers spare parts reference"

    if "controller" in q:
        q += " controller installation replacement assembly function location"

    if "torque" in q:
        q += " torque sensor installation assembly replacement function"

    if "motor" in q or "powertrain" in q or "electric system" in q or "electric support" in q or "no support" in q:
        q += " motor powertrain installation replacement troubleshooting battery controller connector electric support no assist"

    if "rack" in q:
        q += " rack installation assembly"

    if "rear light" in q or "taillight" in q or "tail light" in q or "light is blinking" in q:
        q += " rear light installation assembly blinking light controller"

    if "error" in q or "blinking" in q:
        q += " troubleshooting error code blinking"

    if "turn on" in q or "turn off" in q:
        q += " bike does not turn on turns off troubleshooting battery display cable connection"

    if "difference between bafang and mivice" in q or "difference between old and new" in q:
        q += " compare display system old new bafang mivice"

    if "which system" in q or "identify" in q:
        q += " identify old new bafang mivice display d101 dpe180 e181"

    return q


def infer_filter(question: str, forced_system: Optional[str]) -> Optional[str]:
    q = normalize_question(question)
    filters = []

    requested_doc_type = detect_doc_type_from_text(q)
    requested_component = detect_component_from_text(q)
    requested_system = forced_system or detect_system_from_text(q)

    broad_compare = is_identification_question(q) or (
        "difference between bafang and mivice" in q
        or "difference between old and new" in q
        or "difference between the older and newer" in q
    )

    definition_mode = is_definition_question(q) and not is_repair_or_install_question(q)

    if requested_system == "Bafang":
        filters.append("system_supplier eq 'Bafang'")
    elif requested_system == "Mivice":
        filters.append("system_supplier eq 'Mivice'")
    elif requested_system == "Both":
        filters.append("system_supplier eq 'Both'")

    if requested_doc_type and not broad_compare:
        filters.append(f"doc_type eq '{requested_doc_type}'")

    if requested_component and not broad_compare:
        filters.append(f"search.ismatch('{requested_component}', 'component')")

    if definition_mode and requested_component in DEFINITION_COMPONENTS:
        filters = [f for f in filters if not f.startswith("doc_type eq 'assembly'")]

    return " and ".join(filters) if filters else None


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_chunks(question: str, forced_system: Optional[str], top_k: int = 12) -> List[Dict]:
    expanded_question = expand_query(question)
    question_vector = embed_text(expanded_question)
    filter_expr = infer_filter(question, forced_system)

    vector_query = VectorizedQuery(
        vector=question_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector"
    )

    results = search_client.search(
        search_text=expanded_question,
        vector_queries=[vector_query],
        filter=filter_expr,
        top=top_k,
        select=[
            "content",
            "source_file",
            "product",
            "system_supplier",
            "system_generation",
            "component",
            "doc_type",
            "compatibility",
            "manual_version",
            "audience",
            "drivetrain_type",
            "chunk_id",
            "source_type",
            "has_speech",
            "has_subtitles",
            "is_continuation",
            "related_video",
        ],
    )

    return list(results)


def chunk_matches_component(chunk: Dict, question: str) -> bool:
    q = normalize_question(question)
    source = (chunk.get("source_file") or "").lower()
    content = (chunk.get("content") or "").lower()
    component = (chunk.get("component") or "").lower()
    doc_type = (chunk.get("doc_type") or "").lower()

    broad_compare = is_identification_question(q) or (
        "difference between bafang and mivice" in q
        or "difference between old and new" in q
        or "difference between the older and newer" in q
    )
    if broad_compare:
        return True

    if is_definition_question(q) and not is_repair_or_install_question(q):
        requested_component = detect_component_from_text(q)
        if requested_component in DEFINITION_COMPONENTS:
            if doc_type in {"system", "manual"}:
                return True
            if requested_component in source or requested_component in component or requested_component.replace("-", " ") in content:
                return True
            return False

    if is_warranty_question(q):
        return (
            "warranty" in source
            or "warranty" in component
            or "warranty" in doc_type
            or "warranty" in content
            or "claim" in content
        )

    if is_electrical_symptom_question(q):
        return (
            component in {"powertrain", "battery", "controller", "display", "rear-light", "app"}
            or "powertrain" in content
            or "battery" in content
            or "controller" in content
            or "motor" in content
            or "display" in content
            or "light" in content
        )

    if "controller" in q:
        return "controller" in source or "controller" in component or "controller" in content

    if "torque" in q:
        return "torque" in source or "torque-sensor" in component or "torque" in content

    if "display" in q or "screen" in q or "hmi" in q:
        return (
            "display" in source
            or "display" in component
            or "display" in content
            or "screen" in content
            or "hmi" in content
        )

    if "app" in q or "pair" in q or "bluetooth" in q:
        return "app" in source or "app" in component or "bluetooth" in content or "pair" in content

    if "rack" in q:
        return "rack" in source or "rack" in component or "rack" in content

    return True


def filter_chunks_for_question(chunks: List[Dict], question: str) -> List[Dict]:
    filtered = [c for c in chunks if chunk_matches_component(c, question)]
    return filtered if filtered else chunks


def compute_chunk_score(chunk: Dict, question: str, forced_system: Optional[str], user_type: str) -> int:
    q = normalize_question(question)
    score = 0

    source_file = (chunk.get("source_file") or "").lower()
    content = (chunk.get("content") or "").lower()
    component = (chunk.get("component") or "").lower()
    doc_type = (chunk.get("doc_type") or "").lower()
    audience = (chunk.get("audience") or "").lower()
    system_supplier = (chunk.get("system_supplier") or "").lower()

    requested_component = detect_component_from_text(q)
    requested_doc_type = detect_doc_type_from_text(q)
    requested_system = forced_system or detect_system_from_text(q)

    definition_mode = is_definition_question(q) and not is_repair_or_install_question(q)
    broad_compare = is_identification_question(q) or (
        "difference between bafang and mivice" in q
        or "difference between old and new" in q
        or "difference between the older and newer" in q
    )

    if requested_component and requested_component == component:
        score += 8
    elif requested_component and requested_component in source_file:
        score += 5

    if requested_doc_type and requested_doc_type == doc_type:
        score += 6

    if requested_system and requested_system.lower() == system_supplier:
        score += 7
    elif requested_system == "Both" and system_supplier == "both":
        score += 7

    if user_type == "dealer" and audience == "dealer":
        score += 4
    if user_type == "customer" and audience == "customer":
        score += 4

    if is_warranty_question(q):
        if doc_type == "warranty":
            score += 12
        if "warranty" in source_file:
            score += 10
        if audience == "dealer":
            score += 6

    if definition_mode:
        if doc_type == "system":
            score += 10
        if doc_type == "manual":
            score += 7
        if doc_type == "assembly":
            score -= 6

    if broad_compare:
        if "display" in source_file or "display" in component:
            score += 8
        if doc_type == "manual":
            score += 3
        if system_supplier in {"bafang", "mivice", "both"}:
            score += 3

    if any(x in q for x in BOTH_COMPONENTS) and system_supplier == "both":
        score += 8

    if "controller" in q and is_location_question(q):
        if "headtube" in content or "head tube" in content:
            score += 10
        if doc_type == "assembly":
            score += 4

    if is_electrical_symptom_question(q):
        if component in {"powertrain", "battery", "controller", "display", "rear-light"}:
            score += 10
        if doc_type == "troubleshooting":
            score += 8
        if "motor" in content or "battery" in content or "controller" in content:
            score += 4
        if component == "stem":
            score -= 12

    return score


def rerank_chunks(chunks: List[Dict], question: str, forced_system: Optional[str], user_type: str) -> List[Dict]:
    return sorted(
        chunks,
        key=lambda c: compute_chunk_score(c, question, forced_system, user_type),
        reverse=True
    )


def print_debug_chunks(label: str, chunks: List[Dict]):
    if not DEBUG_RETRIEVAL:
        return

    print(f"\nDEBUG {label}:")
    if not chunks:
        print("  (no chunks)")
        return

    for i, c in enumerate(chunks, start=1):
        print(
            f"  {i}. "
            f"source_file={c.get('source_file')} | "
            f"source_type={c.get('source_type')} | "
            f"component={c.get('component')} | "
            f"doc_type={c.get('doc_type')} | "
            f"system_supplier={c.get('system_supplier')} | "
            f"audience={c.get('audience')}"
        )
    print()


def build_context(chunks: List[Dict], limit: int = 6) -> str:
    parts = []
    for i, doc in enumerate(chunks[:limit], start=1):
        parts.append(
            f"""[Chunk {i}]
source_file: {doc.get("source_file")}
source_type: {doc.get("source_type")}
product: {doc.get("product")}
system_supplier: {doc.get("system_supplier")}
system_generation: {doc.get("system_generation")}
component: {doc.get("component")}
doc_type: {doc.get("doc_type")}
compatibility: {doc.get("compatibility")}
manual_version: {doc.get("manual_version")}
audience: {doc.get("audience")}
drivetrain_type: {doc.get("drivetrain_type")}
chunk_id: {doc.get("chunk_id")}
has_speech: {doc.get("has_speech")}
has_subtitles: {doc.get("has_subtitles")}
is_continuation: {doc.get("is_continuation")}
related_video: {doc.get("related_video")}
content:
{doc.get("content")}
"""
        )
    return "\n\n".join(parts)


def has_meaningful_context(question: str, chunks: List[Dict]) -> bool:
    q = normalize_question(question)
    if not chunks:
        return False

    broad_compare = is_identification_question(q) or (
        "difference between bafang and mivice" in q
        or "difference between old and new" in q
        or "difference between the older and newer" in q
    )
    if broad_compare:
        return True

    if is_warranty_question(q):
        return any(
            "warranty" in " ".join([
                str(c.get("source_file") or "").lower(),
                str(c.get("content") or "").lower(),
                str(c.get("component") or "").lower(),
                str(c.get("doc_type") or "").lower(),
            ])
            for c in chunks[:5]
        )

    strong_terms = []
    component = detect_component_from_text(q)
    doc_type = detect_doc_type_from_text(q)

    if component:
        strong_terms.append(component.replace("-", " "))
    if doc_type:
        strong_terms.append(doc_type)

    if not strong_terms:
        return True

    for chunk in chunks[:5]:
        haystack = " ".join([
            str(chunk.get("source_file") or "").lower(),
            str(chunk.get("content") or "").lower(),
            str(chunk.get("component") or "").lower(),
            str(chunk.get("doc_type") or "").lower(),
            str(chunk.get("system_supplier") or "").lower(),
        ])
        if any(term in haystack for term in strong_terms):
            return True

    return False


# =========================================================
# ANSWERING
# =========================================================

def generate_answer(question: str, context: str, technical_mode: bool, user_type: str, forced_system: Optional[str]) -> str:
    mode_instruction = (
        "This is a dealer-style or technical question. Operational detail is allowed if it is clearly supported by the context."
        if technical_mode or user_type == "dealer"
        else
        "This is a customer-style question. Keep it clear, practical, and easy to understand."
    )

    system_note = (
        f"The identified system is: {forced_system}."
        if forced_system else
        "No system has been explicitly identified yet."
    )

    system_prompt = f"""You are an Ahooga after-sales assistant.

STRICT RULES:
- ONLY answer using the provided context
- If the answer is not clearly supported by the context, say you do not have enough information
- NEVER invent procedures, specifications, coverage terms, comparison claims, or support channels
- NEVER mix Bafang and Mivice guidance
- NEVER mix old and new MAX generations
- NEVER expose dealer-only instructions to a customer unless the context clearly supports a customer-safe version
- Do not say 'contact support' or 'contact your dealer' unless it is truly necessary as a last step
- Do not mention sources, files, chunks, retrieval, or internal metadata
- Do not offer official resources outside the provided context
- Do not ask a follow-up question unless it is truly needed

BEHAVIOR:
- Answer directly first
- Use short paragraphs or bullets only when it improves clarity
- If the question is about identification, comparison, warranty, or a common definition, explain it in natural language
- If the question is broad and the context supports it, summarize instead of refusing
- If the answer is partly supported but not complete, say what you know and where the limit is
- For definition questions, explain the part or concept first before mentioning procedures
- If a component belongs to both systems according to context, do not force a supplier distinction
- For warranty questions, keep the answer focused on the retailer claim process if the context supports that
- Never drift to app features or unrelated product information
- For troubleshooting questions, stay on the most likely subsystem and do not jump to unrelated mechanical parts

QUESTION MODE:
- {mode_instruction}
- {system_note}

STYLE:
- Sound like a competent human after-sales assistant
- Be concise but useful
- Avoid robotic repetition
- Avoid filler
"""

    response = openai_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
User question:
{question}

Retrieved context:
{context}

Write the best possible answer using ONLY the retrieved context.
If the answer is partially supported, answer the supported part clearly and state the remaining limit briefly.
"""
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


# =========================================================
# OUTPUT SANITIZING
# =========================================================

def sanitize_output(answer: str) -> str:
    if not answer:
        return "I do not have enough reliable information in the current knowledge base to answer that properly."

    blocked_fragments = [
        "open connection manager link",
        "verify credentials",
        "retry or cancel the request",
        "explanation_of_tool_call",
        "new_instruction",
        "contentfiltered conversation id",
        "responsible ai restrictions",
        "error message:",
        "conversation id:",
        "connectorrequestfailure",
        "http error with code 500",
    ]
    lower_answer = answer.lower()

    if any(fragment in lower_answer for fragment in blocked_fragments):
        return "I do not have enough reliable information in the current knowledge base to answer that properly."

    stripped = answer.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return "I do not have enough reliable information in the current knowledge base to answer that properly."

    answer = answer.replace("Ahooga support", "after-sales support")
    answer = answer.replace("feel free to ask", "")
    answer = answer.replace("Feel free to ask", "")
    answer = answer.replace("handy ", "")
    answer = answer.strip()

    while "  " in answer:
        answer = answer.replace("  ", " ")

    return answer


# =========================================================
# MAIN ENTRYPOINT
# =========================================================

def answer_question(user_input: str, conversation_id: Optional[str] = None) -> str:
    user_input = user_input.strip()
    if not user_input:
        return "Please provide a question."

    q_norm = normalize_question(user_input)

    # Rewrite / simplify follow-up
    if is_rewrite_followup(q_norm):
        previous_answer = get_last_answer(conversation_id)
        if previous_answer:
            rewritten = rewrite_previous_answer(user_input, previous_answer)
            final = sanitize_output(rewritten)
            remember_turn(
                conversation_id=conversation_id,
                user_question=user_input,
                answer=final,
                forced_system=get_last_system(conversation_id),
                component=get_last_component(conversation_id),
            )
            return final

    # Short contextual follow-up
    expanded_followup = expand_followup_question(user_input, conversation_id)
    if expanded_followup:
        return answer_question(expanded_followup, conversation_id=conversation_id)

    # Standalone system follow-up
    if is_standalone_system_reply(q_norm):
        pending_question = get_pending_question(conversation_id)
        if pending_question:
            clear_pending_question(conversation_id)
            merged = f"{pending_question} ({q_norm})"
            return answer_question(merged, conversation_id=conversation_id)

        system = detect_system_from_text(q_norm)
        if system == "Bafang":
            return "Understood — you mean the older MAX with the Bafang system. Please send the full question in one message."
        if system == "Mivice":
            return "Understood — you mean the newer MAX with the Mivice system. Please send the full question in one message."
        return "Please send the full question together with the bike or system details."

    # Fast-path definitions
    if is_simple_definition(q_norm):
        definition_answer = generate_definition_answer(q_norm)
        if definition_answer:
            final = sanitize_output(definition_answer)
            remember_turn(
                conversation_id=conversation_id,
                user_question=user_input,
                answer=final,
                forced_system=detect_system_from_text(q_norm),
                component=detect_component_from_text(q_norm),
            )
            return final

    # Fast-path comparisons / identification
    if "difference between bafang and mivice" in q_norm or "what is the difference between bafang and mivice" in q_norm:
        final = sanitize_output(answer_compare_systems())
        remember_turn(
            conversation_id=conversation_id,
            user_question=user_input,
            answer=final,
            forced_system=None,
            component="display",
        )
        return final

    if (
        "difference between old and new" in q_norm
        or "difference between the older and newer max systems" in q_norm
        or "how do i identify if my max has the old or new system" in q_norm
        or "how do i know if my max has the old or new system" in q_norm
        or "which system do i have" in q_norm
        or "how do i identify if my max is old or new" in q_norm
        or "how do i identify if my max is old or new?" in q_norm
    ):
        final = sanitize_output(answer_identify_old_or_new())
        remember_turn(
            conversation_id=conversation_id,
            user_question=user_input,
            answer=final,
            forced_system=None,
            component="display",
        )
        return final

    forced_system = detect_system_from_text(q_norm)
    user_type = detect_user_type(q_norm)
    technical_mode = is_technical_question(q_norm)
    component = detect_component_from_text(q_norm)

    if is_return_question(q_norm):
        final = sanitize_output(answer_return_question())
        remember_turn(conversation_id, user_input, final, forced_system, component)
        return final

    if is_location_question(q_norm) and "controller" in q_norm and forced_system:
        final = sanitize_output(answer_controller_location_fallback(forced_system))
        remember_turn(conversation_id, user_input, final, forced_system, "controller")
        return final

    if is_electrical_symptom_question(q_norm) and not forced_system:
        remember_pending_question(conversation_id, user_input)
        final = sanitize_output(answer_identification_clarifier())
        remember_turn(conversation_id, user_input, final, None, component or "powertrain")
        return final

    if needs_system_clarification(q_norm, forced_system):
        remember_pending_question(conversation_id, user_input)
        final = sanitize_output(answer_identification_clarifier())
        remember_turn(conversation_id, user_input, final, None, component)
        return final

    chunks = retrieve_chunks(q_norm, forced_system, top_k=12)
    print_debug_chunks("RETRIEVED BEFORE FILTER", chunks)

    if not chunks:
        if is_location_question(q_norm) and "controller" in q_norm:
            final = sanitize_output(answer_controller_location_fallback(forced_system))
            remember_turn(conversation_id, user_input, final, forced_system, "controller")
            return final

        if is_vague_power_issue(q_norm):
            final = sanitize_output(answer_vague_power_issue(q_norm, forced_system))
            remember_turn(conversation_id, user_input, final, forced_system, component)
            return final

        return "I do not have enough reliable information in the current knowledge base to answer that properly."

    chunks = filter_chunks_for_question(chunks, q_norm)
    chunks = rerank_chunks(chunks, q_norm, forced_system, user_type)
    print_debug_chunks("RETRIEVED AFTER FILTER + RERANK", chunks)

    if not has_meaningful_context(q_norm, chunks):
        if is_location_question(q_norm) and "controller" in q_norm:
            final = sanitize_output(answer_controller_location_fallback(forced_system))
            remember_turn(conversation_id, user_input, final, forced_system, "controller")
            return final

        if is_vague_power_issue(q_norm):
            final = sanitize_output(answer_vague_power_issue(q_norm, forced_system))
            remember_turn(conversation_id, user_input, final, forced_system, component)
            return final

        # softer fallback: try with weaker context instead of failing hard
        context = build_context(chunks, limit=3)
        answer = generate_answer(q_norm, context, technical_mode, user_type, forced_system)
        final = sanitize_output(answer)
        remember_turn(conversation_id, user_input, final, forced_system, component)
        return final

    context = build_context(chunks, limit=6)
    answer = generate_answer(q_norm, context, technical_mode, user_type, forced_system)
    final = sanitize_output(answer)

    remember_turn(
        conversation_id=conversation_id,
        user_question=user_input,
        answer=final,
        forced_system=forced_system,
        component=component,
    )
    return final


# =========================================================
# CHAT LOOP
# =========================================================

def chat():
    print("Ahooga AI Assistant ready. Type 'exit' to quit.\n")
    conversation_id = "local_cli"

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            break

        answer = answer_question(user_input, conversation_id=conversation_id)

        print("\nAssistant:\n")
        print(answer)
        print("\n---\n")


if __name__ == "__main__":
    chat()