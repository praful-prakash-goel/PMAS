import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import styles from "../styles/alerts.module.css";

// fields from class diagram: AlertId, MachineId, Severity, Timestamp, Acknowledged
const initialAlerts = [
  { AlertId: "ALT-001", MachineId: "M-103", TicketId: "TKT-1247", Severity: "critical", Timestamp: "2 min ago",   Acknowledged: false },
  { AlertId: "ALT-002", MachineId: "M-207", TicketId: "TKT-1248", Severity: "critical", Timestamp: "15 min ago",  Acknowledged: false },
  { AlertId: "ALT-003", MachineId: "M-045", TicketId: "TKT-1245", Severity: "warning",  Timestamp: "1 hour ago",  Acknowledged: false },
  { AlertId: "ALT-004", MachineId: "M-156", TicketId: "TKT-1240", Severity: "warning",  Timestamp: "2 hours ago", Acknowledged: true  },
  { AlertId: "ALT-005", MachineId: "M-031", TicketId: "TKT-1238", Severity: "critical", Timestamp: "3 hours ago", Acknowledged: true  },
  { AlertId: "ALT-006", MachineId: "M-012", TicketId: "TKT-1235", Severity: "warning",  Timestamp: "5 hours ago", Acknowledged: true  },
];

const severityOrder = { critical: 1, warning: 2 };

const Badge = ({ severity }) => {
  const cls = severity === "critical" ? styles.criticalBadge : styles.warningBadge;
  return (
    <span className={`${styles.badge} ${cls}`}>
      <span className={styles.bdot} />
      {severity.charAt(0).toUpperCase() + severity.slice(1)}
    </span>
  );
};

const Alerts = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const role = new URLSearchParams(location.search).get("user");
  const isAdmin = role === "administrator";

  const [alerts, setAlerts] = useState(initialAlerts);
  const [filter, setFilter] = useState("all");
  const [notifSent, setNotifSent] = useState(null);

  const filtered = alerts
    .filter((a) => {
      if (filter === "unacknowledged") return !a.Acknowledged;
      if (filter === "acknowledged")   return a.Acknowledged;
      return true;
    })
    .sort((a, b) => {
      if (a.Acknowledged !== b.Acknowledged) return a.Acknowledged ? 1 : -1;
      return severityOrder[a.Severity] - severityOrder[b.Severity];
    });

  const acknowledge = (AlertId) => {
    // TODO: POST /alerts/:AlertId/acknowledge — backend sends notif to all technicians
    setAlerts((prev) =>
      prev.map((a) => a.AlertId === AlertId ? { ...a, Acknowledged: true } : a)
    );
    setNotifSent(AlertId);
    setTimeout(() => setNotifSent(null), 3000);
  };

  return (
    <div className={styles.page}>

      <div className={styles.topBar}>
        <div>
          <div className={styles.pageTitle}>Alerts</div>
          <div className={styles.pageSubtitle}>
            {alerts.filter((a) => !a.Acknowledged).length} unacknowledged alerts
          </div>
        </div>
        <select className={styles.select} value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="all">All Alerts</option>
          <option value="unacknowledged">Unacknowledged</option>
          <option value="acknowledged">Acknowledged</option>
        </select>
      </div>

      {notifSent && (
        <div className={styles.toast}>
          ✅ Alert {notifSent} acknowledged — notification sent to all technicians
        </div>
      )}

      <div className={styles.grid}>
        {filtered.map((alert) => (
          <div
            key={alert.AlertId}
            className={`${styles.card} ${styles[alert.Severity]} ${alert.Acknowledged ? styles.acked : ""}`}
          >
            <div className={styles.cardTop}>
              <span className={styles.alertId}>{alert.AlertId}</span>
              <Badge severity={alert.Severity} />
            </div>

            <div className={styles.machineId}>{alert.MachineId}</div>

            <div className={styles.ticketRow}>
              <span className={styles.ticketId}>#{alert.TicketId}</span>
              <button
                className={styles.viewTicketBtn}
                onClick={() => navigate(`/tickets${location.search}`)}
              >
                View Ticket →
              </button>
            </div>

            <div className={styles.timestamp}>🕐 {alert.Timestamp}</div>

            <div className={styles.cardFooter}>
              {!alert.Acknowledged ? (
                isAdmin && (
                  <button
                    className={styles.acknowledgeBtn}
                    onClick={() => acknowledge(alert.AlertId)}
                  >
                    Acknowledge
                  </button>
                )
              ) : (
                <span className={styles.ackedText}>✓ Acknowledged — Technicians notified</span>
              )}
            </div>
          </div>
        ))}
        {filtered.length === 0 && (
          <p className={styles.empty}>No alerts found.</p>
        )}
      </div>

    </div>
  );
};

export default Alerts;