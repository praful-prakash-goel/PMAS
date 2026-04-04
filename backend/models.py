import enum
from sqlalchemy import Column, Integer, String, Boolean, Date, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database import Base

class UserRole(enum.Enum):
    ADMIN = "ADMIN"
    TECHNICIAN = "TECHNICIAN"

class HealthStatus(enum.Enum):
    HEALTHY = "HEALTHY"
    DEGRADING = "DEGRADING"
    CRITICAL = "CRITICAL"

class Severity(enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class TicketStatus(enum.Enum):
    OPEN = "OPEN"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN PROGRESS"
    RESOLVED = "RESOLVED"

class Priority(enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), nullable=False)
    org_name = Column(String(100))

    # relationships
    technician = relationship(
        "Technician", 
        back_populates="user", 
        uselist=False,
        passive_deletes=True, 
        cascade="all, delete-orphan" 
    )

class Technician(Base):
    __tablename__ = "technicians"

    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    availability_status = Column(Boolean, default=True)

    # relationships
    user = relationship("User", back_populates="technician")
    tickets = relationship("TicketTechnician", back_populates="technician")
    
    @property
    def assigned_tickets(self):
        return [link.ticket for link in self.tickets if link.ticket]

class Machine(Base):
    __tablename__ = "machines"

    machine_id = Column(String(50), primary_key=True, index=True)
    machine_type = Column(String(100))
    health_status = Column(Enum(HealthStatus))
    installation_date = Column(Date)
    last_service_date = Column(Date)
    location = Column(String(100))
    org_name = Column(String(100))

    # relationships
    alerts = relationship("Alert", back_populates="machine", cascade="all, delete")

class Alert(Base):
    __tablename__ = "alerts"

    alert_id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String(50), ForeignKey("machines.machine_id", ondelete="CASCADE"))
    severity = Column(Enum(Severity))
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged = Column(Boolean, default=False)
    closed = Column(Boolean, default=False)

    # relationships
    machine = relationship("Machine", back_populates="alerts")
    tickets = relationship("Ticket", back_populates="alert", cascade="all, delete")

class Ticket(Base):
    __tablename__ = "tickets"

    ticket_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    alert_id = Column(Integer, ForeignKey("alerts.alert_id", ondelete="CASCADE"))
    priority = Column(Enum(Priority))
    status = Column(
        Enum(TicketStatus, values_callable=lambda obj: [e.value for e in obj]),
        default=TicketStatus.OPEN
    )
    created_at = Column(DateTime, default=datetime.utcnow)

    # relationships
    alert = relationship("Alert", back_populates="tickets")
    technicians = relationship("TicketTechnician", back_populates="ticket")
    
    @property
    def accepted_by(self):
        return [link.technician.user for link in self.technicians]

    @property
    def assigned_to(self):
        return [link.technician.user for link in self.technicians if link.is_assigned]
    
    @property
    def machine_id(self):
        return self.alert.machine_id if self.alert else None

class TicketTechnician(Base):
    __tablename__ = "ticket_technicians"

    ticket_id = Column(Integer, ForeignKey("tickets.ticket_id", ondelete="CASCADE"), primary_key=True)
    technician_id = Column(Integer, ForeignKey("technicians.user_id", ondelete="CASCADE"), primary_key=True)
    is_assigned = Column(Boolean, default=False)

    # relationships
    ticket = relationship("Ticket", back_populates="technicians")
    technician = relationship("Technician", back_populates="tickets")