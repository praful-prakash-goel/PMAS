from typing import List
from fastapi import APIRouter, status, Depends, HTTPException, Body
from backend import database, auth, models, schemas
from backend.core.hashing import Hash
from sqlalchemy.orm import Session

router = APIRouter(
    prefix='/technician',
    tags=['technician']
)
get_db = database.get_db
role_required = auth.role_required

@router.patch('/update-info')
def update_technician_info(
    request: schemas.UserUpdate,
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['TECHNICIAN']))
):
    updated_info = request.dict(exclude_unset=True)
    if "old_password" not in updated_info:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Old password is required to update information.")
    
    if not Hash.verify(updated_info["old_password"], user.password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Old password does not match")
    
    if "new_password" in updated_info:
        user.password = Hash.bcrypt(updated_info["new_password"])
        updated_info.pop("new_password")
    
    updated_info.pop("old_password", None)
    for key, value in updated_info.items():
        setattr(user, key, value)
    
    db.commit()
    db.refresh(user)
    
    return user

@router.get('/machines', response_model=List[schemas.MachineResponse])
def get_all_machines(
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['TECHNICIAN']))
):
    machines = db.query(models.Machine).filter(
        models.Machine.org_name == user.org_name
    ).all()
    
    return machines

@router.get('/my-tickets', response_model=List[schemas.TicketResponse])
def get_assigned_tickets(
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['TECHNICIAN']))
):
    ticket_records = db.query(models.TicketTechnician).filter(
        models.TicketTechnician.technician_id == user.user_id,
        models.TicketTechnician.is_assigned == True
    ).all()
    
    ticket_ids = [r.ticket_id for r in ticket_records]
    
    my_tickets = db.query(models.Ticket).filter(
        models.Ticket.ticket_id.in_(ticket_ids)
    ).all()
    
    return my_tickets
    
@router.get('/open-tickets', response_model=List[schemas.TicketResponse])
def get_open_tickets(
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['TECHNICIAN']))
):
    open_tickets = db.query(models.Ticket).filter(
        models.Ticket.status == models.TicketStatus.OPEN
    ).all()
    
    return open_tickets

@router.post('/accept-ticket/{ticket_id}')
def accept_ticket(
    ticket_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['TECHNICIAN']))
):
    open_ticket = db.query(models.Ticket).filter(
        models.Ticket.ticket_id == ticket_id,
        models.Ticket.status == models.TicketStatus.OPEN
    ).first()
    
    if not open_ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Ticket {ticket_id} not found")
    
    existing = db.query(models.TicketTechnician).filter(
        models.TicketTechnician.ticket_id == ticket_id,
        models.TicketTechnician.technician_id == user.user_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="You have already applied for this ticket")
    
    new_ticket_record = models.TicketTechnician(
        ticket_id=ticket_id,
        technician_id=user.user_id
    )
    
    db.add(new_ticket_record)
    db.commit()
    db.refresh(new_ticket_record)
    
    return {"msg": f"Ticket {ticket_id} accepted by user {user.user_id}"}

@router.patch('/update-status/{ticket_id}', response_model=schemas.TicketResponse)
def update_ticket_status(
    ticket_id: int,
    status: models.TicketStatus = Body(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['TECHNICIAN']))
):
    assignment = db.query(models.TicketTechnician).filter(
        models.TicketTechnician.ticket_id == ticket_id,
        models.TicketTechnician.technician_id == user.user_id,
        models.TicketTechnician.is_assigned == True
    ).first()
    
    if not assignment:
        raise HTTPException(
            status_code=403, 
            detail="You are not authorized to update this ticket's status."
        )
        
    ticket = db.query(models.Ticket).filter(
        models.Ticket.ticket_id == ticket_id
    ).first()
    
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Ticket {ticket_id} not found")
    
    ticket.status = status.value
    
    db.commit()
    
    return ticket