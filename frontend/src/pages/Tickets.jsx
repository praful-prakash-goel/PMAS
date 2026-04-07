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
  const userId = localStorage.getItem("userId"); // Current logged in ID
  const isAdmin = role === "ADMIN";

  const [adminTickets, setAdminTickets] = useState([]);
  const [myTickets, setMyTickets] = useState([]);
  const [openTickets, setOpenTickets] = useState([]);
  const [loading, setLoading] = useState(true);
  
  const [statusFilter, setStatusFilter] = useState("all");
  const [priorityFilter, setPriorityFilter] = useState("all"); 
  const [showAssignModal, setShowAssignModal] = useState(false);
  const [applicants, setApplicants] = useState([]); 
  const [selectedTicketId, setSelectedTicketId] = useState(null);
  const [selectedTechIds, setSelectedTechIds] = useState([]);

  const fetchTickets = async () => {
    try {
      if (isAdmin) {
        const res = await fetch(`${BASE_URL}/admin/tickets`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json();
        setAdminTickets(Array.isArray(data) ? data : []);
      } else {
        // Calling the specific technician endpoints
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

        setMyTickets(Array.isArray(myData) ? myData : []);
        setOpenTickets(Array.isArray(openData) ? openData : []);
      }
    } catch (err) {
      console.error("Fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const fetchTickets = async () => {
      setLoading(true);
      try {
        // Only fetch technician-specific data
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

        // Ensure we are setting arrays to avoid .map() errors
        setMyTickets(Array.isArray(myData) ? myData : []);
        setOpenTickets(Array.isArray(openData) ? openData : []);
        
        console.log("My Tickets Loaded:", myData);
        console.log("Open Tickets Loaded:", openData);
      } catch (err) {
        console.error("Fetch error:", err);
        setMyTickets([]);
        setOpenTickets([]);
      } finally {
        setLoading(false);
      }
    };

    if (token && !isAdmin) {
      fetchTickets();
    } else if (token && isAdmin) {
      // Logic for Admin fetch remains same as your original
      const fetchAdminTickets = async () => {
        const res = await fetch(`${BASE_URL}/admin/tickets`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json();
        setAdminTickets(Array.isArray(data) ? data : []);
        setLoading(false);
      };
      fetchAdminTickets();
    }
  }, [token, isAdmin]);

  const filterLogic = (list) => {
    return list.filter((t) => {
      const matchesStatus = statusFilter === "all" || t.status === statusFilter;
      const matchesPriority = priorityFilter === "all" || t.priority === priorityFilter;
      return matchesStatus && matchesPriority;
    });
  };

  const handleAccept = async (ticketId) => {
    try {
      const res = await fetch(`${BASE_URL}/technician/accept-ticket/${ticketId}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      
      if (res.ok) {
        alert("Wait for admin to confirm assignment.");
        
        // Update local state immediately so the button shifts to 'Pending Approval'
        setOpenTickets((prev) =>
          prev.map((t) => {
            if (t.ticket_id === ticketId) {
              // We simulate the backend junction table entry
              const newApplicant = { 
                technician_id: Number(userId), 
                is_assigned: false 
              };
              return { 
                ...t, 
                technicians: [...(t.technicians || []), newApplicant] 
              };
            }
            return t;
          })
        );
      } else {
        const err = await res.json();
        alert(err.detail || "You have already applied for this ticket.");
      }
    } catch (err) {
      console.error("Something went wrong with the acceptance:", err);
    }
  };

  const updateStatus = async (ticketId, newStatus) => {
    try {
      const res = await fetch(`${BASE_URL}/technician/update-status/${ticketId}`, {
        method: "PATCH",
        headers: { 
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify(newStatus), // Sending string "RESOLVED"
      });
      if (res.ok) {
        fetchTickets();
      }
    } catch {
        console.error("Failed to push status update to server.");
    }
  };

  const openAssignModal = (ticket) => {
    //Identify the list of technicians who hit 'Accept'
    const applicantsList = ticket.technicians || ticket.accepted_by || [];

    if (applicantsList.length > 0) {
      // Format the data for the Modal UI
      // We map through the TicketTechnician junction table to reach the User data
      const formattedApplicants = applicantsList.map(item => ({
        // This handles both flat and nested structures
        user_id: item.technician?.user?.user_id || item.technician_id || item.user_id,
        name: item.technician?.user?.name || item.name || `Technician ${item.technician_id}`,
        email: item.technician?.user?.email || item.email || ""
      }));

      setApplicants(formattedApplicants);
      setSelectedTicketId(ticket.ticket_id);
      setShowAssignModal(true);
      
      // Identify who is already assigned (is_assigned: True in DB)
      // This ensures the checkboxes start 'checked' for already assigned techs
      const alreadyAssigned = applicantsList
        .filter(item => item.is_assigned === true)
        .map(item => item.technician_id);
        
      setSelectedTechIds(alreadyAssigned);
    } else {
      // If this shows up, check the console to see the structure of 'ticket'
      console.log("Empty Applicants for Ticket:", ticket);
      alert("No technicians have accepted this ticket yet.");
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
        // alert("Assignment successful!");
        setShowAssignModal(false);
        fetchTickets();
      }
    } catch (err) {
      console.error("Assignment error:", err);
    }
  };

  // Helper to render a ticket card
  const renderTicketCard = (t) => {
    const currentUserId = Number(userId);

    //Check 'accepted_by' array to see if you applied
    const hasApplied = t.accepted_by?.some(
      (tech) => Number(tech.user_id) === currentUserId
    );

    //Check 'assigned_to' array to see if Admin assigned it to you
    const isAssignedToMe = t.assigned_to?.some(
      (tech) => Number(tech.user_id) === currentUserId
    );

    //Logic: If ticket is assigned but NOT to me, hide it from the technician view
    if (!isAdmin && t.status === "ASSIGNED" && !isAssignedToMe) {
      return null;
    }

    return (
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
          <span className={styles.created}>
            {/* Using time_passed from your JSON for a better look */}
            {t.time_passed || new Date(t.created_at).toLocaleString()}
          </span>
        </div>

        <div className={styles.cardFooter}>
          {isAdmin ? (
            <button 
              className={styles.assignBtn} 
              onClick={() => openAssignModal(t)}
              disabled={t.status !== "OPEN"} 
              style={t.status !== "OPEN" ? { opacity: 0.5, cursor: 'not-allowed' } : {}}
            >
              {t.status === "OPEN" ? "Assign Technicians ▾" : "Already Assigned"}
            </button>
          ) : (
            <>
              {/* OPEN STATE */}
              {t.status === "OPEN" && (
                hasApplied ? (
                  <button className={styles.pendingBtn} disabled>Pending Approval</button>
                ) : (
                  <button className={styles.startBtn} onClick={() => handleAccept(t.ticket_id)}>Accept Ticket</button>
                )
              )}

              {/* ASSIGNED STATE */}
              {t.status === "ASSIGNED" && isAssignedToMe && (
                <button className={styles.resolveBtn} onClick={() => updateStatus(t.ticket_id, "RESOLVED")}>
                  ✓ Resolve
                </button>
              )}

              {/* RESOLVED STATE */}
              {t.status === "RESOLVED" && (
                <span className={styles.doneText}>Completed</span>
              )}
            </>
          )}
        </div>
      </div>
    );
  };

  // if (loading) return <div className={styles.loading}>Pulling latest tickets...</div>;

  return (
    <div className={styles.page}>
      <div className={styles.topBar}>
        <div>
          <div className={styles.pageTitle}>Tickets</div>
          {/* <div className={styles.pageSubtitle}>System Maintenance Tasks</div> */}
        </div>
      </div>

      <div className={styles.filterSection} style={{ display: "flex", gap: "20px", marginBottom: "20px" }}>
        <select className={styles.select} value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
          <option value="all">All Statuses ▾</option>
          <option value="OPEN">Open</option>
          <option value="ASSIGNED">Assigned</option>
          <option value="RESOLVED">Resolved</option>
        </select>
        <select className={styles.select} value={priorityFilter} onChange={(e) => setPriorityFilter(e.target.value)}>
          <option value="all">All Priorities ▾</option>
          <option value="HIGH">High</option>
          <option value="MEDIUM">Medium</option>
          <option value="LOW">Low</option>
        </select>
      </div>

      {isAdmin ? (
        <div className={styles.grid}>{filterLogic(adminTickets).map(renderTicketCard)}</div>
      ) : (
        <>
          <h3 className={styles.sectionTitle}>My Tickets</h3>
          <div className={styles.grid} style={{ marginBottom: '40px' }}>
            {filterLogic(myTickets).map(renderTicketCard)}
            {filterLogic(myTickets).length === 0 && <p className={styles.empty}>No assigned tickets yet.</p>}
          </div>

          <h3 className={styles.sectionTitle}>Open Tickets</h3>
          <div className={styles.grid}>
            {filterLogic(openTickets).map(renderTicketCard)}
            {filterLogic(openTickets).length === 0 && <p className={styles.empty}>No open tickets available.</p>}
          </div>
        </>
      )}

      {showAssignModal && (
        <div className={styles.overlay} onClick={() => setShowAssignModal(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3>Assign Technicians for #TKT-{selectedTicketId}</h3>
              <button className={styles.closeBtn} onClick={() => setShowAssignModal(false)}>✕</button>
            </div>
            <div className={styles.form} style={{ padding: '20px' }}>
              <p>Select technicians:</p>
              <div className={styles.techSelectionList}>
                {applicants.map((tech) => (
                  <label key={tech.user_id} style={{ display: 'flex', gap: '10px', marginBottom: '10px', cursor: 'pointer' }}>
                    <input 
                      type="checkbox" 
                      checked={selectedTechIds.includes(tech.user_id)}
                      onChange={(e) => {
                        const id = tech.user_id;
                        setSelectedTechIds(prev => e.target.checked ? [...prev, id] : prev.filter(i => i !== id));
                      }}
                    />
                    {tech.name} ({tech.email})
                  </label>
                ))}
              </div>
              <div className={styles.modalActions} style={{ marginTop: '20px' }}>
                <button type="button" className={styles.cancelBtn} onClick={() => setShowAssignModal(false)}>Cancel</button>
                <button type="button" className={styles.addBtn} onClick={handleFinalAssignment}>Confirm Assignment</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Tickets;