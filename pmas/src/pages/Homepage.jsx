import { useLocation } from "react-router-dom";
import styles from "../styles/homepage.module.css";

const CURRENT_USER = "John Smith";

const allTickets = [
  { id: "#TKT-1247", machine: "M-103", issue: "Replace cooling system",   priority: "critical", status: "open",        assignee: "John Smith"    },
  { id: "#TKT-1240", machine: "M-156", issue: "Conveyor belt alignment",  priority: "low",      status: "open",        assignee: "John Smith"    },
  { id: "#TKT-1238", machine: "M-031", issue: "Unusual noise from motor", priority: "high",     status: "in-progress", assignee: "John Smith"    },
  { id: "#TKT-1248", machine: "M-207", issue: "Inspect vibration sensor", priority: "high",     status: "in-progress", assignee: "Sarah Johnson" },
  { id: "#TKT-1245", machine: "M-045", issue: "Routine maintenance",      priority: "medium",   status: "in-progress", assignee: "Mike Chen"     },
  { id: "#TKT-1235", machine: "M-012", issue: "Lubrication check",        priority: "low",      status: "resolved",    assignee: "John Smith"    },
];

const allMachines = [
  { id: "M-103", name: "CNC Mill",        location: "Production Line A", status: "critical" },
  { id: "M-207", name: "Lathe",           location: "Warehouse B",       status: "critical" },
  { id: "M-045", name: "Press",           location: "Assembly Floor",    status: "warning"  },
  { id: "M-156", name: "Conveyor Belt",   location: "Production Line C", status: "warning"  },
  { id: "M-012", name: "CNC Lathe",       location: "Production Line A", status: "healthy"  },
  { id: "M-089", name: "Drill Press",     location: "Warehouse B",       status: "healthy"  },
  { id: "M-031", name: "Milling Machine", location: "Production Line B", status: "healthy"  },
  { id: "M-074", name: "Band Saw",        location: "Assembly Floor",    status: "healthy"  },
];

const allTechnicians = [
  { name: "John Smith",    email: "john@senseact.com",  assigned: 3, status: "active"   },
  { name: "Sarah Johnson", email: "sarah@senseact.com", assigned: 2, status: "active"   },
  { name: "Mike Chen",     email: "mike@senseact.com",  assigned: 1, status: "inactive" },
];

const statColors = {
  open:       "#e05252",
  inprogress: "#e0a052",
  resolved:   "#52a876",
  total:      "#263A99",
  critical:   "#e05252",
  warning:    "#e0a052",
  healthy:    "#52a876",
};

const HomePage = () => {
  const location = useLocation();
  const role = new URLSearchParams(location.search).get("user");
  const isAdmin = role === "administrator";

  const ticketSource = isAdmin ? allTickets : allTickets.filter((t) => t.assignee === CURRENT_USER);
  const open       = ticketSource.filter((t) => t.status === "open").length;
  const inProgress = ticketSource.filter((t) => t.status === "in-progress").length;
  const resolved   = ticketSource.filter((t) => t.status === "resolved").length;

  const critical = allMachines.filter((m) => m.status === "critical").length;
  const warning  = allMachines.filter((m) => m.status === "warning").length;
  const healthy  = allMachines.filter((m) => m.status === "healthy").length;

  const myMachineIds = [...new Set(ticketSource.map((t) => t.machine))];
  const myMachines = allMachines.filter((m) => myMachineIds.includes(m.id));
  const attentionMachines = allMachines.filter((m) => m.status !== "healthy");

  return (
    <div className={styles.page}>

      {/* Header */}
      <div className={styles.header}>
        <div>
          <h2 className={styles.title}>Welcome back, {isAdmin ? "Admin" : CURRENT_USER.split(" ")[0]}</h2>
          <p className={styles.sub}>Here's your shift overview</p>
        </div>
      </div>

      {/* Stat strip */}
      <div className={styles.statStrip}>
        <div className={styles.statItem} style={{ borderTopColor: statColors.open }}>
          <span className={styles.statNum}>{open}</span>
          <span className={styles.statLbl}>Open Tickets</span>
        </div>
        <div className={styles.statItem} style={{ borderTopColor: statColors.inprogress }}>
          <span className={styles.statNum}>{inProgress}</span>
          <span className={styles.statLbl}>In Progress</span>
        </div>
        <div className={styles.statItem} style={{ borderTopColor: statColors.resolved }}>
          <span className={styles.statNum}>{resolved}</span>
          <span className={styles.statLbl}>Resolved</span>
        </div>
        <div className={styles.statItem} style={{ borderTopColor: statColors.critical }}>
          <span className={styles.statNum}>{critical}</span>
          <span className={styles.statLbl}>Critical Machines</span>
        </div>
        <div className={styles.statItem} style={{ borderTopColor: statColors.warning }}>
          <span className={styles.statNum}>{warning}</span>
          <span className={styles.statLbl}>Warning</span>
        </div>
        <div className={styles.statItem} style={{ borderTopColor: statColors.healthy }}>
          <span className={styles.statNum}>{healthy}</span>
          <span className={styles.statLbl}>Healthy</span>
        </div>
      </div>

      {/* Main grid */}
      <div className={styles.grid}>

        {/* Left: machines */}
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>
            {isAdmin ? "Machines Needing Attention" : "My Assigned Machines"}
          </h3>
          <div className={styles.list}>
            {(isAdmin ? attentionMachines : myMachines).map((m) => (
              <div key={m.id} className={styles.listRow}>
                <div className={styles.listLeft}>
                  <span className={styles.listId}>{m.id}</span>
                  <span className={styles.listName}>{m.name}</span>
                </div>
                <div className={styles.listRight}>
                  <span className={styles.listSub}>{m.location}</span>
                  <span className={`${styles.badge} ${styles[m.status + "Badge"]}`}>
                    {m.status.charAt(0).toUpperCase() + m.status.slice(1)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: tickets or technicians */}
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>
            {isAdmin ? "Technicians" : "My Tickets"}
          </h3>
          <div className={styles.list}>
            {isAdmin
              ? allTechnicians.map((t) => (
                  <div key={t.name} className={styles.listRow}>
                    <div className={styles.listLeft}>
                      <span className={styles.listName}>{t.name}</span>
                      <span className={styles.listSub}>{t.email}</span>
                    </div>
                    <div className={styles.listRight}>
                      <span className={styles.listSub}>{t.assigned} assigned</span>
                      <span className={`${styles.badge} ${t.status === "active" ? styles.healthyBadge : styles.warningBadge}`}>
                        {t.status.charAt(0).toUpperCase() + t.status.slice(1)}
                      </span>
                    </div>
                  </div>
                ))
              : ticketSource.map((t) => (
                  <div key={t.id} className={styles.listRow}>
                    <div className={styles.listLeft}>
                      <span className={styles.listId}>{t.id}</span>
                      <span className={styles.listName}>{t.issue}</span>
                    </div>
                    <div className={styles.listRight}>
                      <span className={`${styles.badge} ${styles[t.priority + "Badge"]}`}>
                        {t.priority.charAt(0).toUpperCase() + t.priority.slice(1)}
                      </span>
                      <span className={`${styles.badge} ${styles[t.status.replace("-", "") + "Status"]}`}>
                        {t.status.charAt(0).toUpperCase() + t.status.slice(1).replace("-", " ")}
                      </span>
                    </div>
                  </div>
                ))
            }
          </div>
        </div>

      </div>
    </div>
  );
};

export default HomePage;