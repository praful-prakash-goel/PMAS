from datetime import datetime, date, timezone
from pydantic import BaseModel, computed_field, field_validator
from backend.models import UserRole, HealthStatus, Priority, TicketStatus, Severity
from typing import List, Optional

class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    role: UserRole
    org_name: str
    
class UserLogin(BaseModel):
    email: str
    password: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    old_password: Optional[str] = None
    new_password: Optional[str] = None
    org_name: Optional[str] = None
    
class TicketCompact(BaseModel):
    ticket_id: int
    priority: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True
        
class TechnicianProfile(BaseModel):
    availability_status: bool
    assigned_tickets: List[TicketCompact] = []

    class Config:
        from_attributes = True
    
class UserResponse(BaseModel):
    name: str
    email: str
    org_name: str
    technician: Optional[TechnicianProfile]

    class Config:
        from_attributes = True
    
class MachineCreate(BaseModel):
    machine_type: str
    installation_date: date
    location: str
    org_name: str
    
    @field_validator("installation_date", mode="before")
    @classmethod
    def parse_dd_mm_yyyy(cls, value):
        clean_value = value.strip()
        if isinstance(value, str):
            try:
                return datetime.strptime(clean_value, "%d-%m-%Y").date()
            except:
                raise ValueError(f"Invalid format {value}. Format must be DD-MM-YYYY (e.g., 21-03-2026)")
        return value

class MachineUpdate(BaseModel):
    health_status: Optional[HealthStatus] = None
    location: Optional[str] = None
    last_service_date: Optional[date] = None
    
    @field_validator("last_service_date", mode="before")
    @classmethod
    def parse_dd_mm_yyyy(cls, value):
        clean_value = value.strip()
        if isinstance(value, str):
            try:
                return datetime.strptime(clean_value, "%d-%m-%Y").date()
            except:
                raise ValueError(f"Invalid format {value}. Expected DD-MM-YYYY (e.g., 21-03-2026)")
        return value

class MachineResponse(BaseModel):
    machine_id: str
    machine_type: str
    health_status: str
    installation_date: date
    last_service_date: date
    location: str
    org_name: str
    
    class Config:
        from_attributes = True
    
class TicketCreate(BaseModel):
    alert_id: int
    priority: Optional[Priority] = None
    status: TicketStatus
    
class TechnicianSimple(BaseModel):
    user_id: int
    name: str
    email: str

    class Config:
        from_attributes = True
    
class TicketResponse(BaseModel):
    ticket_id: int
    alert_id: int
    machine_id: str
    priority: Priority
    status: TicketStatus
    created_at: datetime
    
    accepted_by: List[TechnicianSimple] = []
    assigned_to: List[TechnicianSimple] = []
    
    @computed_field
    @property
    def time_passed(self) -> str:
        now = datetime.now(timezone.utc)
        diff = now - self.created_at.replace(tzinfo=timezone.utc)
        
        days = diff.days
        hours = diff.seconds // 3600
        minutes = (diff.seconds // 60) % 60
        
        if days > 0:
            return f"{days}d {hours}h ago"
        if hours > 0:
            return f"{hours}h {minutes}m ago"
        return f"{minutes}m ago"

    class Config:
        from_attributes = True

class AlertResponse(BaseModel):
    alert_id: int
    machine_id: str
    severity: Severity
    created_at: datetime
    acknowledged: bool
    
    @computed_field
    @property
    def time_passed(self) -> str:
        now = datetime.now(timezone.utc)
        diff = now - self.created_at.replace(tzinfo=timezone.utc)
        
        days = diff.days
        hours = diff.seconds // 3600
        minutes = (diff.seconds // 60) % 60
        
        if days > 0:
            return f"{days}d {hours}h ago"
        if hours > 0:
            return f"{hours}h {minutes}m ago"
        return f"{minutes}m ago"

    class Config:
        from_attributes=True