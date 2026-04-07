import { useState, useEffect } from "react";
// import { useNavigate } from "react-router-dom";
import styles from "../styles/alerts.module.css";

const BASE_URL = "http://localhost:8000";

/* // fields from class diagram: AlertId, MachineId, Severity, Timestamp, Acknowledged
const initialAlerts = [
  { AlertId: "ALT-001", MachineId: "M-103", TicketId: "TKT-1247", Severity: "critical", Timestamp: "2 min ago",   Acknowledged: false },
  { AlertId: "ALT-002", MachineId: "M-207", TicketId: "TKT-1248", Severity: "critical", Timestamp: "15 min ago",  Acknowledged: false },
  { AlertId: "ALT-003", MachineId: "M-045", TicketId: "TKT-1245", Severity: "warning",  Timestamp: "1 hour ago",  Acknowledged: false },
  { AlertId: "ALT-004", MachineId: "M-156", TicketId: "TKT-1240", Severity: "warning",  Timestamp: "2 hours ago", Acknowledged: true  },
  { AlertId: "ALT-005", MachineId: "M-031", TicketId: "TKT-1238", Severity: "critical", Timestamp: "3 hours ago", Acknowledged: true  },
  { AlertId: "ALT-006", MachineId: "M-012", TicketId: "TKT-1235", Severity: "warning",  Timestamp: "5 hours ago", Acknowledged: true  },
];

const severityOrder = { critical: 1, warning: 2 };
*/

const Badge = ({ severity }) => {
  // DB might return 'HIGH' or 'critical' - normalizing for the CSS classes
  const s = severity?.toUpperCase();
  const map = {
    // CRITICAL: styles.badgeCritical,
    // WARNING: styles.badgeWarning,
    HIGH: styles.badgeHigh,
    MEDIUM: styles.badgeMedium,
    LOW: styles.badgeLow,
  };
  
  return (
    <span className={`${styles.badge} ${map[s] || ""}`}>
      <span className={styles.bdot} />
      {severity?.charAt(0).toUpperCase() + severity?.slice(1).toLowerCase()}
    </span>
  );
};

const Alerts = () => {
  // const navigate = useNavigate();
  const token = localStorage.getItem("token");
  
  // Using localStorage for role check to match your login system
  const userRole = localStorage.getItem("userRole");
  const isAdmin = userRole === "ADMIN";

  const [alerts, setAlerts] = useState([]); // Switched from initialAlerts to empty array for API
  // const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all");
  const [notifSent, setNotifSent] = useState(null);

  // Pulling live data from the server
  useEffect(() => {
    fetchAlerts();
  }, []);

  const fetchAlerts = async () => {
    try {
      const res = await fetch(`${BASE_URL}/admin/alerts`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      // Safety check: ensure we got an array back
      setAlerts(Array.isArray(data) ? data : []);
    } catch {
      console.error("Fetch failed. Backend might be down.");
    } finally {
      // setLoading(false);
    }
  };

  /* // Old local-only acknowledge logic
  const acknowledge = (AlertId) => {
    setAlerts((prev) =>
      prev.map((a) => a.AlertId === AlertId ? { ...a, Acknowledged: true } : a)
    );
    setNotifSent(AlertId);
    setTimeout(() => setNotifSent(null), 3000);
  };
  */

  const acknowledge = async (alert) => {
    // This now handles the DB update AND the ticket creation in one go
    try {
      const aId = alert.alert_id;
      
      const ackRes = await fetch(`${BASE_URL}/admin/alerts/${aId}/acknowledge`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (ackRes.ok) {
        // Automatically spinning up a ticket for this alert
        const ticketRes = await fetch(`${BASE_URL}/admin/tickets`, {
          method: "POST",
          headers: { 
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json" 
          },
          body: JSON.stringify({
            alert_id: aId,
            priority: alert.severity, 
            status: "OPEN" 
          }),
        });

        if (ticketRes.ok) {
          setNotifSent(aId);
          fetchAlerts(); // Refreshing list to show the '✓ Acknowledged' state
          setTimeout(() => setNotifSent(null), 3000);
        }
      }
    } catch (err) {
      console.error("Process failed:", err);
    }
  };

  const filtered = (alerts || []).filter((a) => {
    if (filter === "unacknowledged") return !a.acknowledged;
    if (filter === "acknowledged") return a.acknowledged;
    return true;
  });

  // if (loading) return <div className={styles.loading}>Connecting to alert system...</div>;

  return (
    <div className={styles.page}>

      {/* Header Section */}
      <div className={styles.topBar}>
        <div>
          <div className={styles.pageTitle}>Alerts</div>
          <div className={styles.pageSubtitle}>
            {alerts.filter((a) => !a.acknowledged).length} unacknowledged alerts
          </div>
        </div>
      </div>

      {/* Filter Section shifted below heading to match other tabs */}
      <div className={styles.filterSection} style={{ marginBottom: "20px" }}>
        <select className={styles.select} value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="all">All Alerts ▾</option>
          <option value="unacknowledged">Unacknowledged ▾</option>
          <option value="acknowledged">Acknowledged ▾</option>
        </select>
      </div>

      {notifSent && (
        <div className={styles.toast}>
          Alert {notifSent} acknowledged — Ticket created successfully!
        </div>
      )}

      <div className={styles.grid}>
        {filtered.map((alert) => {
          // Logic to find the ticket ID from the relationship array in your model
          const linkedTicket = alert.tickets && alert.tickets.length > 0 
            ? alert.tickets[0].ticket_id 
            : null;

          console.log(linkedTicket);

          return (
            <div
              key={alert.alert_id}
              className={`${styles.card} ${styles[alert.severity?.toLowerCase()]} ${alert.acknowledged ? styles.acked : ""}`}
            >
              <div className={styles.cardTop}>
                <span className={styles.alertId}>#ALT-{alert.alert_id}</span>
                <Badge severity={alert.severity} />
              </div>

              <div className={styles.machineId}>Machine ID: {alert.machine_id}</div>

              {/* Ticket ID row mapping to backend relationship */}
              <div className={styles.ticketRow}>
                <span className={styles.ticketId}>
                   {linkedTicket ? `#TKT-${linkedTicket}` : "Ticket: PENDING"}
                </span>
              </div>

              {/* Created At display mapping to SQLAlchemy created_at */}
              <div className={styles.timestamp}>
                 Created At: {alert.created_at ? new Date(alert.created_at).toLocaleString() : "Recent"}
              </div>

              <div className={styles.cardFooter}>
                {!alert.acknowledged ? (
                  isAdmin && (
                    <button
                      className={styles.acknowledgeBtn}
                      onClick={() => acknowledge(alert)}
                    >
                      Acknowledge
                    </button>
                  )
                ) : (
                  <span className={styles.ackedText}>✓ Ticket Created</span>
                )}
              </div>
            </div>
          );
        })}
        {filtered.length === 0 && (
          <p className={styles.empty}>No alerts found.</p>
        )}
      </div>

    </div>
  );
};

export default Alerts;