import { useState, useEffect } from "react";
// import { useLocation } from "react-router-dom";
import styles from "../styles/tickets.module.css";

const BASE_URL = "http://localhost:8000";

// const CURRENT_USER = "John Smith";

// TODO: fetch from GET /technicians
// const allTechnicians = [
//   { id: 1, name: "John Smith",      status: "active"   },
//   { id: 2, name: "Sarah Johnson", status: "active"   },
//   { id: 3, name: "Mike Chen",       status: "inactive" },
// ];

// const activeTechs = allTechnicians.filter((t) => t.status === "active");

// // fields from class diagram: TicketId, AlertId, Priority, Status, AssignedTo, Created
// const initialTickets = [
//   { TicketId: "TKT-1247", AlertId: "ALT-001", machine: "M-103", issue: "Replace cooling system",   Priority: "critical", Status: "open",       AssignedTo: ["John Smith"],    Created: "2 min ago"   },
//   { TicketId: "TKT-1240", AlertId: "ALT-002", machine: "M-156", issue: "Conveyor belt alignment",  Priority: "low",       Status: "open",       AssignedTo: ["John Smith"],    Created: "3 hours ago" },
//   { TicketId: "TKT-1238", AlertId: "ALT-003", machine: "M-031", issue: "Unusual noise from motor", Priority: "high",      Status: "inprogress", AssignedTo: ["John Smith"],    Created: "5 hours ago" },
//   { TicketId: "TKT-1248", AlertId: "ALT-004", machine: "M-207", issue: "Inspect vibration sensor", Priority: "high",      Status: "inprogress", AssignedTo: ["Sarah Johnson"], Created: "15 min ago"  },
//   { TicketId: "TKT-1245", AlertId: "ALT-005", machine: "M-045", issue: "Routine maintenance",       Priority: "medium",    Status: "inprogress", AssignedTo: ["Mike Chen"],     Created: "1 hour ago"  },
//   { TicketId: "TKT-1235", AlertId: "ALT-006", machine: "M-012", issue: "Lubrication check",         Priority: "low",       Status: "resolved",   AssignedTo: ["John Smith"],    Created: "1 day ago"   },
// ];

// const priorityOrder = { critical: 1, high: 2, medium: 3, low: 4 };

const Badge = ({ status }) => {
  const map = {
    CRITICAL: styles.badgeCritical,
    OPEN: styles.badgeOpen,
    ASSIGNED: styles.badgeInprogress,
    RESOLVED: styles.badgeResolved,
    HIGH: styles.badgeHigh,
    MEDIUM: styles.badgeMedium,
    LOW: styles.badgeLow,
  };
  
  const s = status?.toUpperCase();
  const lbl = { ASSIGNED: "Assigned", OPEN: "Open" };
  
  return (
    <span className={`${styles.badge} ${map[s] || ""}`}>
      <span className={styles.bdot} />
      {lbl[s] || status?.charAt(0).toUpperCase() + status?.slice(1).toLowerCase()}
    </span>
  );
};

const Tickets = () => {
  const role = localStorage.getItem("userRole"); 
  const token = localStorage.getItem("token");
  const userId = localStorage.getItem("userId");
  const isAdmin = role === "ADMIN";

  const [tickets, setTickets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState("all");
  const [priorityFilter, setPriorityFilter] = useState("all"); 
  const [showAssignModal, setShowAssignModal] = useState(false);
  const [applicants, setApplicants] = useState([]); 
  const [selectedTicketId, setSelectedTicketId] = useState(null);
  const [selectedTechIds, setSelectedTechIds] = useState([]);
  // const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    const fetchTickets = async () => {
      try {
        if (isAdmin) {
          const res = await fetch(`${BASE_URL}/admin/tickets`, {
            headers: { Authorization: `Bearer ${token}` },
          });
          const data = await res.json();
          setTickets(Array.isArray(data) ? data : []);
        } else {
          const [myRes, openRes] = await Promise.all([
            fetch(`${BASE_URL}/technician/my-tickets`, {
              headers: { Authorization: `Bearer ${token}` },
            }),
            fetch(`${BASE_URL}/technician/open-tickets`, {
              headers: { Authorization: `Bearer ${token}` },
            })
          ]);

          const myData = await myRes.json();
          const openData = await openRes.json();

          setTickets([...(Array.isArray(myData) ? myData : []), ...(Array.isArray(openData) ? openData : [])]);
        }
      } catch (err) {
        console.error("Fetch error:", err);
        setTickets([]); 
      } finally {
        setLoading(false);
      }
    };
    if (token) fetchTickets();
  }, [token, isAdmin]);

  const filtered = tickets.filter((t) => {
    const matchesStatus = statusFilter === "all" || t.status === statusFilter;
    const matchesPriority = priorityFilter === "all" || t.priority === priorityFilter;
    
    if (isAdmin) return matchesStatus && matchesPriority;

    const isAvailable = t.status === "OPEN";
    const isMine = t.status === "ASSIGNED" && t.assigned_technicians?.includes(Number(userId));
    
    return matchesStatus && matchesPriority && (isAvailable || isMine);
  });

  const handleAccept = async (ticketId) => {
    try {
      const res = await fetch(`${BASE_URL}/technician/accept-ticket/${ticketId}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) alert("Applied! Wait for admin to confirm assignment.");
    } catch {
      alert("Something went wrong with the acceptance.");
    }
  };

  const updateStatus = async (TicketId, newStatus) => {
    try {
      const res = await fetch(`${BASE_URL}/technician/update-status/${TicketId}`, {
        method: "PATCH",
        headers: { 
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify(newStatus),
      });
      if (res.ok) {
        setTickets(tickets.map((t) => t.ticket_id === TicketId ? { ...t, status: newStatus } : t));
      }
    } catch {
        console.error("Failed to push status update to server.");
    }
  };

  const openAssignModal = async (ticket) => {
    try {
      // This helper fetch would get technicians who hit 'Accept'
      const res = await fetch(`${BASE_URL}/admin/tickets/${ticket.ticket_id}/applicants`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();

      if (res.ok && data.length > 0) {
        setApplicants(data);
        setSelectedTicketId(ticket.ticket_id);
        setShowAssignModal(true);
        setSelectedTechIds([]); 
      } else {
        alert("No technicians have applied for this ticket yet.");
      }
    } catch (err) {
      console.error("Error fetching applicants:", err);
    }
  };

  const handleFinalAssignment = async () => {
    if (selectedTechIds.length === 0) return alert("Select at least one technician.");

    try {
      const res = await fetch(`${BASE_URL}/admin/tickets/${selectedTicketId}/assign`, {
        method: "PATCH",
        headers: { 
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json" 
        },
        body: JSON.stringify(selectedTechIds), 
      });

      if (res.ok) {
        alert("Assignment successful!");
        setShowAssignModal(false);
        window.location.reload(); 
      } else {
        const err = await res.json();
        alert(err.detail || "Assignment failed.");
      }
    } catch (err) {
      console.error("Assignment error:", err);
    }
  };

  if (loading) return <div className={styles.loading}>Pulling latest tickets...</div>;

  return (
    <div className={styles.page}>
      <div className={styles.topBar}>
        <div>
          <div className={styles.pageTitle}>Tickets</div>
          <div className={styles.pageSubtitle}>{filtered.length} active work items</div>
        </div>
      </div>

      {/* Filters Section - Shifted below the heading */}
      <div className={styles.filterSection} style={{ display: "flex", gap: "20px", marginBottom: "20px", alignItems: "center" }}>
        <div className={styles.filterGroup}>
          <select className={styles.select} value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="all">All Statuses ▾</option>
            <option value="OPEN">Open</option>
            <option value="ASSIGNED">Assigned</option>
            <option value="RESOLVED">Resolved</option>
          </select>
        </div>

        <div className={styles.filterGroup}>
          <select className={styles.select} value={priorityFilter} onChange={(e) => setPriorityFilter(e.target.value)}>
            <option value="all">All Priorities ▾</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>
        </div>
      </div>

      <div className={styles.grid}>
        {filtered.map((t) => (
          <div key={t.ticket_id} className={styles.card}>
            <div className={styles.cardTop}>
              <span className={styles.ticketId}>#TKT-{t.ticket_id}</span>
              <Badge status={t.status} />
            </div>

            <div className={styles.meta}>
              <span className={styles.machine}>Machine: {t.machine_id || "Unspecified"}</span>
              <span className={styles.alertId}>Alert Id: {t.alert_id}</span>
            </div>

            <div className={styles.cardMid}>
              <Badge status={t.priority} />
              {/* Created At label with date/time */}
              <span className={styles.created}>Created At: {t.created_at}</span>
            </div>

            <div className={styles.cardFooter}>
              {isAdmin ? (
                <button className={styles.assignBtn} onClick={() => openAssignModal(t)}>
                  Assign Technicians ▾
                </button>
              ) : (
                <>
                  {t.status === "OPEN" && (
                    <button className={styles.startBtn} onClick={() => handleAccept(t.ticket_id)}>
                      Accept Ticket
                    </button>
                  )}
                  {t.status === "ASSIGNED" && (
                    <button className={styles.resolveBtn} onClick={() => updateStatus(t.ticket_id, "RESOLVED")}>✓ Resolve</button>
                  )}
                  {t.status === "RESOLVED" && (
                    <span className={styles.doneText}>✓ Completed</span>
                  )}
                </>
              )}
            </div>
          </div>
        ))}
        {filtered.length === 0 && <p className={styles.empty}>Nothing to show here.</p>}
      </div>

      {showAssignModal && (
        <div className={styles.overlay} onClick={() => setShowAssignModal(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3>Assign Technicians for #TKT-{selectedTicketId}</h3>
              <button className={styles.closeBtn} onClick={() => setShowAssignModal(false)}>✕</button>
            </div>
            
            <div className={styles.form} style={{ padding: '20px' }}>
              <p style={{ marginBottom: '15px' }}>Select from technicians who accepted this ticket:</p>
              
              <div className={styles.techSelectionList}>
                {applicants.map((tech) => (
                  <label key={tech.user_id} style={{ display: 'flex', gap: '10px', marginBottom: '10px', cursor: 'pointer' }}>
                    <input 
                      type="checkbox" 
                      checked={selectedTechIds.includes(tech.user_id)}
                      onChange={(e) => {
                        const id = tech.user_id;
                        setSelectedTechIds(prev => 
                          e.target.checked ? [...prev, id] : prev.filter(i => i !== id)
                        );
                      }}
                    />
                    {tech.name} (Email: {tech.email})
                  </label>
                ))}
              </div>

              <div className={styles.modalActions} style={{ marginTop: '20px' }}>
                <button type="button" className={styles.cancelBtn} onClick={() => setShowAssignModal(false)}>
                  Cancel
                </button>
                <button type="button" className={styles.addBtn} onClick={handleFinalAssignment}>
                  Confirm Assignment
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Tickets;