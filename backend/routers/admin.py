from typing import List
from fastapi import APIRouter, status, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc
from backend import database, schemas, auth, models
from backend.core.hashing import Hash

router = APIRouter(
    prefix='/admin',
    tags=['admin']
)
get_db = database.get_db
get_current_user = auth.get_current_user
role_required = auth.role_required

@router.patch('/update-info')
def update_admin_info(
    request: schemas.UserUpdate,
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['ADMIN']))
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

@router.get('/technicians', response_model=List[schemas.UserResponse])
def get_all_technicians(
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['ADMIN']))
):
    technicians = db.query(models.User).filter(
        models.User.role == models.UserRole.TECHNICIAN,
        models.User.org_name == user.org_name
    ).all()
    
    return technicians

@router.post('/technicians', status_code=status.HTTP_201_CREATED)
def register_technician(
    request: schemas.UserCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['ADMIN']))
):
    hashed_password = Hash.bcrypt(request.password)
    
    new_user = models.User(
        name=request.name,
        email=request.email,
        password=hashed_password,
        org_name=request.org_name,
        role=models.UserRole.TECHNICIAN
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    new_technician = models.Technician(
        user_id=new_user.user_id,
        availability_status=True
    )
    
    db.add(new_technician)
    db.commit()
    
    return {"msg": "Technician created successfully"}

@router.delete('/technicians/{technician_email}')
def remove_technician(
    technician_email: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['ADMIN']))
):
    technician = db.query(models.User).filter(models.User.email == technician_email).first()
    
    if not technician:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Technician not found")

    if not technician.org_name == user.org_name:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="You cannot delete technicians from other organizations")
    
    db.delete(technician)
    db.commit()
    return {
        "msg": f"Technician {technician_email} deleted successfully"
    }


@router.get('/machines', response_model=List[schemas.MachineResponse])
def get_all_machines(
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['ADMIN']))
):
    machines = db.query(models.Machine).filter(
        models.Machine.org_name == user.org_name
    ).all()
    
    return machines

@router.post('/machines')
def add_machine(
    request: schemas.MachineCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['ADMIN']))
):
    last_machine = db.query(models.Machine)\
                    .order_by(desc(models.Machine.machine_id))\
                    .limit(1)\
                    .scalar()
    
    if last_machine is None:
        machine_id = 'M01'
    else:
        last_machine_id = last_machine.machine_id
        id_num = int(last_machine_id[1:])
        next_id = id_num + 1
        
        if next_id < 10:
            machine_id = 'M0' + str(next_id)
        else:
            machine_id = 'M' + str(next_id)
    
    machine = models.Machine(
        machine_id=machine_id,
        machine_type=request.machine_type,
        health_status=models.HealthStatus.HEALTHY,
        installation_date=request.installation_date,
        last_service_date=request.installation_date,
        location=request.location,
        org_name=user.org_name
    )
    
    db.add(machine)
    db.commit()
    db.refresh(machine)
    
    return {"msg": "Machine added successfully"}

@router.delete('/machines/{machine_id}')
def remove_machine(
    machine_id: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['ADMIN']))
):
    machine = db.query(models.Machine).filter(models.Machine.machine_id == machine_id).first()
    
    if not machine:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Machine not found")

    if not machine.org_name == user.org_name:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="You cannot delete machines from other organizations")
    
    db.delete(machine)
    db.commit()
    return {
        "msg": f"Machine {machine_id} deleted successfully"
    }

@router.patch('/machine/{machine_id}')
def update_machine_info(
    machine_id: str,
    request: schemas.MachineUpdate,
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['ADMIN']))
):
    updated_info = request.dict(exclude_unset=True)
    
    machine = db.query(models.Machine).filter(models.Machine.machine_id == machine_id).first()
    if not machine:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Machine not found")
    
    for key, value in updated_info.items():
        setattr(machine, key, value)
        
    db.commit()
    db.refresh(machine)
    
    return machine

def get_alert_by_id(db: Session, alert_id: int):
    alert = db.query(models.Alert).filter(
        models.Alert.alert_id == alert_id
    ).first()
    
    return alert

@router.get('/tickets', response_model=List[schemas.TicketResponse])
def get_all_tickets(
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['ADMIN']))
):
    tickets = db.query(models.Ticket).all()
    
    return tickets

@router.post('/tickets')
def create_ticket(
    request: schemas.TicketCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['ADMIN']))
):
    alert = get_alert_by_id(db=db, alert_id=request.alert_id)
    
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Alert {request.alert_id} not found")
    
    if request.priority is not None:
        final_priority = request.priority
    else:
        final_priority = alert.severity.value
        
    new_ticket = models.Ticket(
        alert_id=alert.alert_id,
        priority=final_priority,
        status=request.status
    )
    
    db.add(new_ticket)
    db.commit()
    db.refresh(new_ticket)
    
    return {"msg": "Ticket added succesfully"}

@router.patch('/tickets/{ticket_id}/assign')
def assign_ticket(
    ticket_id: int,
    technician_ids: List[int],
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['ADMIN']))
):
    ticket_records = db.query(models.TicketTechnician).filter(
        models.TicketTechnician.ticket_id == ticket_id
    ).all()
    
    record_lookup = {r.technician_id: r for r in ticket_records}
    
    for record in ticket_records:
        record.is_assigned = False
        
    for id in technician_ids:
        if id not in record_lookup:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Techinican id {id} has not yet accepted the ticket.")
        
        record = record_lookup[id]
        record.is_assigned = True
    
    ticket = db.query(models.Ticket).get(ticket_id)
    ticket.status = models.TicketStatus.ASSIGNED if technician_ids else models.TicketStatus.OPEN
    
    db.commit()
    
    return {"msg": "Ticket assigned succesfully"}

@router.get('/alerts', response_model=List[schemas.AlertResponse])
def view_alerts(
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['ADMIN']))
):
    alerts = db.query(models.Alert).all()
    
    return alerts

@router.get('/alerts/{alert_id}/acknowledge')
def acknowlege_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(role_required(['ADMIN']))
):
    alert = get_alert_by_id(db=db, alert_id=alert_id)
    
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Alert {alert_id} not found")
    
    alert.acknowledged = True
    
    db.commit()
    
    return {"msg": "Alert acknowledged succesfully"}