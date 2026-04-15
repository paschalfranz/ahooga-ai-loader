from services.odoo_client import OdooClient


class TicketService:
    def __init__(self):
        self.client = OdooClient()

    def get_solved_dealer_tickets(self, limit=20):
        domain = [
            ["stage_id.name", "in", ["Solved", "Closed"]],
            ["team_id.name", "=", "Dealers"],
        ]

        fields = [
            "id",
            "name",
            "partner_id",
            "team_id",
            "stage_id",
            "create_date",
            "write_date",
            "description",
        ]

        return self.client.search_read(
            model="helpdesk.ticket",
            domain=domain,
            fields=fields,
            limit=limit,
            order="write_date desc",
        )

    def get_ticket_by_id(self, ticket_id: int):
        records = self.client.read(
            model="helpdesk.ticket",
            ids=[ticket_id],
            fields=[
                "id",
                "name",
                "partner_id",
                "team_id",
                "stage_id",
                "create_date",
                "write_date",
                "description",
                "message_ids",
            ],
        )
        return records[0] if records else None

    def get_ticket_thread(self, ticket_id: int):
        ticket = self.get_ticket_by_id(ticket_id)
        if not ticket:
            return None

        message_ids = ticket.get("message_ids", [])
        if not message_ids:
            return {
                "ticket": ticket,
                "messages": [],
            }

        messages = self.client.read(
            model="mail.message",
            ids=message_ids,
            fields=[
                "id",
                "date",
                "subject",
                "body",
                "author_id",
                "message_type",
                "subtype_id",
                "model",
                "res_id",
            ],
        )

        messages = sorted(messages, key=lambda x: x.get("date") or "")

        return {
            "ticket": ticket,
            "messages": messages,
        }

    def extract_case_from_ticket(self, ticket_id: int):
        thread_data = self.get_ticket_thread(ticket_id)
        if not thread_data:
            return None

        ticket = thread_data["ticket"]
        messages = thread_data["messages"]

        first_message = ""
        final_message = ""

        if messages:
            first_message = messages[0].get("body", "") or ""
            final_message = messages[-1].get("body", "") or ""
        else:
            first_message = ticket.get("description", "") or ""

        return {
            "ticket_id": ticket["id"],
            "ticket_name": ticket.get("name"),
            "team": ticket.get("team_id"),
            "stage": ticket.get("stage_id"),
            "product_model": None,
            "problem": first_message,
            "solution": final_message,
            "parts_used": [],
            "raw_message_count": len(messages),
            "status": "candidate",
        }