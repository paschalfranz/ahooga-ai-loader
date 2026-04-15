from fastapi import APIRouter, HTTPException, Query
from services.ticket_service import TicketService

router = APIRouter(prefix="/tickets", tags=["tickets"])
ticket_service = TicketService()


@router.get("/solved-dealers")
def get_solved_dealer_tickets(limit: int = Query(20, ge=1, le=200)):
    try:
        return ticket_service.get_solved_dealer_tickets(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticket_id}")
def get_ticket(ticket_id: int):
    try:
        ticket = ticket_service.get_ticket_by_id(ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return ticket
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticket_id}/thread")
def get_ticket_thread(ticket_id: int):
    try:
        result = ticket_service.get_ticket_thread(ticket_id)
        if not result:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{ticket_id}/extract-case")
def extract_case(ticket_id: int):
    try:
        result = ticket_service.extract_case_from_ticket(ticket_id)
        if not result:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))