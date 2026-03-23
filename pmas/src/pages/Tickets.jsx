import { useState } from "react";
import { useLocation } from "react-router-dom";
import ticketStyles from "../styles/tickets.module.css";

const CURRENT_USER = "John Smith";

const initialTickets = [
  { id: "#TKT-1247", machine: "M-103", issue: "Replace cooling system",   priority: "critical", status: "open",        assignee: "John Smith",    created: "2 min ago"   },
  { id: "#TKT-1240", machine: "M-156", issue: "Conveyor belt alignment",  priority: "low",      status: "open",        assignee: "John Smith",    created: "3 hours ago" },
  { id: "#TKT-1238", machine: "M-031", issue: "Unusual noise from motor", priority: "high",     status: "in-progress", assignee: "John Smith",    created: "5 hours ago" },
  { id: "#TKT-1248", machine: "M-207", issue: "Inspect vibration sensor", priority: "high",     status: "in-progress", assignee: "Sarah Johnson", created: "15 min ago"  },
  { id: "#TKT-1245", machine: "M-045", issue: "Routine maintenance",      priority: "medium",   status: "in-progress", assignee: "Mike Chen",     created: "1 hour ago"  },
  { id: "#TKT-1235", machine: "M-012", issue: "Lubrication check",        priority: "low",      status: "resolved",    assignee: "John Smith",    created: "1 day ago"   },
];

const priorityOrder = { critical: 1, high: 2, medium: 3, low: 4 };

const Tickets = () => {
  const location = useLocation();
  const role = new URLSearchParams(location.search).get("user");
  const isAdmin = role === "administrator";

  const [tickets, setTickets] = useState(initialTickets);
  const [statusFilter, setStatusFilter] = useState("all");
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState({ machine: "", issue: "", priority: "medium", assignee: "" });

  // admin sees all tickets, technician sees only their own
  const visibleTickets = isAdmin
    ? tickets
    : tickets.filter((t) => t.assignee === CURRENT_USER);

  const filtered = visibleTickets
    .filter((t) => statusFilter === "all" || t.status === statusFilter)
    .sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);

  const updateStatus = (id, newStatus) => {
    setTickets((prev) => prev.map((t) => (t.id === id ? { ...t, status: newStatus } : t)));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // TODO: connect to backend
    const newTicket = {
      id: `#TKT-${Math.floor(1000 + Math.random() * 9000)}`,
      machine: form.machine,
      issue: form.issue,
      priority: form.priority,
      status: "open",
      assignee: form.assignee,
      created: "just now",
    };
    setTickets([newTicket, ...tickets]);
    setForm({ machine: "", issue: "", priority: "medium", assignee: "" });
    setShowModal(false);
  };

  return (
    <div className={ticketStyles.page}>

      <div className={ticketStyles.topBar}>
        <div>
          <h2 className={ticketStyles.pageTitle}>{isAdmin ? "All Tickets" : "My Tickets"}</h2>
          <p className={ticketStyles.subtitle}>
            {isAdmin ? `${filtered.length} tickets total` : `Assigned to ${CURRENT_USER}`}
          </p>
        </div>
        <div className={ticketStyles.actions}>
          <select className={ticketStyles.select} value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="all">All Status</option>
            <option value="open">Open</option>
            <option value="in-progress">In Progress</option>
            <option value="resolved">Resolved</option>
          </select>
          {isAdmin && (
            <button className={ticketStyles.createBtn} onClick={() => setShowModal(true)}>
              + Create Ticket
            </button>
          )}
        </div>
      </div>

      <div className={ticketStyles.tableWrapper}>
        <table className={ticketStyles.table}>
          <thead>
            <tr>
              <th>Ticket ID</th>
              <th>Machine</th>
              <th>Issue</th>
              <th>Priority</th>
              <th>Status</th>
              {isAdmin && <th>Assigned To</th>}
              <th>Created</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((ticket) => (
              <tr key={ticket.id}>
                <td className={ticketStyles.ticketId}>{ticket.id}</td>
                <td>{ticket.machine}</td>
                <td>{ticket.issue}</td>
                <td>
                  <span className={`${ticketStyles.badge} ${ticketStyles[ticket.priority + "Badge"]}`}>
                    {ticket.priority.charAt(0).toUpperCase() + ticket.priority.slice(1)}
                  </span>
                </td>
                <td>
                  <span className={`${ticketStyles.statusBadge} ${ticketStyles[ticket.status.replace("-", "") + "Status"]}`}>
                    {ticket.status.charAt(0).toUpperCase() + ticket.status.slice(1).replace("-", " ")}
                  </span>
                </td>
                {isAdmin && <td>{ticket.assignee}</td>}
                <td className={ticketStyles.created}>{ticket.created}</td>
                <td>
                  {ticket.status === "open" && (
                    <button className={ticketStyles.actionBtn} onClick={() => updateStatus(ticket.id, "in-progress")}>
                      Start
                    </button>
                  )}
                  {ticket.status === "in-progress" && (
                    <button className={`${ticketStyles.actionBtn} ${ticketStyles.resolveBtn}`} onClick={() => updateStatus(ticket.id, "resolved")}>
                      Resolve
                    </button>
                  )}
                  {ticket.status === "resolved" && (
                    <span className={ticketStyles.done}>✓ Done</span>
                  )}
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={isAdmin ? 8 : 7} className={ticketStyles.empty}>No tickets found.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {isAdmin && showModal && (
        <div className={ticketStyles.overlay} onClick={() => setShowModal(false)}>
          <div className={ticketStyles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={ticketStyles.modalHeader}>
              <h3>Create Maintenance Ticket</h3>
              <button className={ticketStyles.closeBtn} onClick={() => setShowModal(false)}>✕</button>
            </div>
            <form onSubmit={handleSubmit} className={ticketStyles.form}>
              <div className={ticketStyles.inputGroup}>
                <label>Machine</label>
                <select value={form.machine} onChange={(e) => setForm({ ...form, machine: e.target.value })} required>
                  <option value="">Select Machine</option>
                  <option value="M-103">M-103 — CNC Mill</option>
                  <option value="M-207">M-207 — Lathe</option>
                  <option value="M-045">M-045 — Press</option>
                  <option value="M-156">M-156 — Conveyor Belt</option>
                  <option value="M-031">M-031 — Milling Machine</option>
                </select>
              </div>
              <div className={ticketStyles.inputGroup}>
                <label>Issue Description</label>
                <textarea rows={3} placeholder="Describe the issue..."
                  value={form.issue} onChange={(e) => setForm({ ...form, issue: e.target.value })} required />
              </div>
              <div className={ticketStyles.formRow}>
                <div className={ticketStyles.inputGroup}>
                  <label>Priority</label>
                  <select value={form.priority} onChange={(e) => setForm({ ...form, priority: e.target.value })} required>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </div>
                <div className={ticketStyles.inputGroup}>
                  <label>Assign To</label>
                  <select value={form.assignee} onChange={(e) => setForm({ ...form, assignee: e.target.value })} required>
                    <option value="">Select Technician</option>
                    <option value="John Smith">John Smith</option>
                    <option value="Sarah Johnson">Sarah Johnson</option>
                    <option value="Mike Chen">Mike Chen</option>
                  </select>
                </div>
              </div>
              <div className={ticketStyles.modalActions}>
                <button type="button" className={ticketStyles.cancelBtn} onClick={() => setShowModal(false)}>Cancel</button>
                <button type="submit" className={ticketStyles.createBtn}>Create Ticket</button>
              </div>
            </form>
          </div>
        </div>
      )}

    </div>
  );
};

export default Tickets;