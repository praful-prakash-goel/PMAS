import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import styles from "../styles/homepage.module.css";

const BASE_URL = "http://localhost:8000";

// const CURRENT_USER = "John Smith";

// const allTickets = [
//   { id: "#TKT-1247", machine: "M-103", issue: "Replace cooling system",   priority: "critical", status: "open",        assignee: "John Smith"    },
//   { id: "#TKT-1240", machine: "M-156", issue: "Conveyor belt alignment",  priority: "low",      status: "open",        assignee: "John Smith"    },
//   { id: "#TKT-1238", machine: "M-031", issue: "Unusual noise from motor", priority: "high",      status: "in-progress", assignee: "John Smith"    },
//   { id: "#TKT-1248", machine: "M-207", issue: "Inspect vibration sensor", priority: "high",      status: "in-progress", assignee: "Sarah Johnson" },
//   { id: "#TKT-1245", machine: "M-045", issue: "Routine maintenance",       priority: "medium",    status: "in-progress", assignee: "Mike Chen"     },
//   { id: "#TKT-1235", machine: "M-012", issue: "Lubrication check",         priority: "low",       status: "resolved",    assignee: "John Smith"    },
// ];

// const machines = [
//   { id: "M-103", name: "CNC Mill",         location: "Production Line A", status: "critical" },
//   { id: "M-207", name: "Lathe",            location: "Warehouse B",       status: "critical" },
//   { id: "M-045", name: "Press",            location: "Assembly Floor",    status: "warning"  },
//   { id: "M-156", name: "Conveyor Belt",    location: "Production Line C", status: "warning"  },
//   { id: "M-012", name: "CNC Lathe",        location: "Production Line A", status: "healthy"  },
//   { id: "M-089", name: "Drill Press",      location: "Warehouse B",       status: "healthy"  },
//   { id: "M-031", name: "Milling Machine", location: "Production Line B", status: "healthy"  },
//   { id: "M-074", name: "Band Saw",         location: "Assembly Floor",    status: "healthy"  },
// ];

// const allTechnicians = [
//   { name: "John Smith",    email: "john@senseact.com",   assigned: 3, status: "active"   },
//   { name: "Sarah Johnson", email: "sarah@senseact.com", assigned: 2, status: "active"   },
//   { name: "Mike Chen",     email: "mike@senseact.com",  assigned: 1, status: "inactive" },
// ];

const statColors = {
  open:       "#e05252",
  inprogress: "#e0a052",
  resolved:   "#52a876",
  total:       "#263A99",
  critical:   "#e05252",
  warning:    "#e0a052",
  healthy:    "#52a876",
  unacknowledged: "#FF6B6B",
  activeTechs: "#D5FF40",    
};

const HomePage = () => {
  const navigate = useNavigate();  

  // Auth & Session stuff
  const role = localStorage.getItem("userRole");
  const userId = localStorage.getItem("userId");
  const userName = localStorage.getItem("userName"); 
  const token = localStorage.getItem("token");

  // Local state for dashboard data
  const [tickets, setTickets] = useState([]);
  const [machines, setMachines] = useState([]);
  const [technicians, setTechnicians] = useState([]);
  const [alerts, setAlerts] = useState([]); 
  // const [loading, setLoading] = useState(true);

  const isAdmin = role === "ADMIN";

  useEffect(() => {
    // Kick user to login if no token found
    if (!token) {
      navigate("/");
      return;
    }

    const fetchData = async () => {
      try {
        const headers = { "Authorization": `Bearer ${token}` };
        const rolePath = role.toLowerCase();

        // Determine the correct ticket path based on the role
        // Admin uses /tickets, Technician uses /my-tickets
        const ticketEndpoint = isAdmin 
          ? `${BASE_URL}/${rolePath}/tickets` 
          : `${BASE_URL}/${rolePath}/my-tickets`;

        // 2. Execute parallel fetches
        const [tktRes, machRes] = await Promise.all([
          fetch(ticketEndpoint, { headers }),
          fetch(`${BASE_URL}/${rolePath}/machines`, { headers })
        ]);

        if (!tktRes.ok || !machRes.ok) {
          throw new Error(`Fetch failed: ${tktRes.status} / ${machRes.status}`);
        }

        const tktData = await tktRes.json();
        const machData = await machRes.json();

        // update state with a safety check for arrays
        setTickets(Array.isArray(tktData) ? tktData : []);
        setMachines(Array.isArray(machData) ? machData : []);

        //Handle extra Admin-only data if necessary
        if (isAdmin) {
          const techRes = await fetch(`${BASE_URL}/admin/technicians`, { headers });
          if (techRes.ok) setTechnicians(await techRes.json());
          
          const alertRes = await fetch(`${BASE_URL}/admin/alerts`, { headers });
          if (alertRes.ok) setAlerts(await alertRes.json());
        }
      } catch (err) {
        console.error("Dashboard data sync failed:", err);
      } finally {
        // setLoading(false);
      }
    };

    fetchData();
  }, [token, navigate, isAdmin, role]);

  // if (loading) return <div className={styles.loading}>Spinning up dashboard...</div>;

  // Handle cross-check for assigned tickets (Tech view)
  const ticketSource = isAdmin 
    ? tickets 
    : tickets.filter((t) => 
        t.technicians?.some(rel => String(rel.technician_id) === String(userId) && rel.is_assigned) ||
        t.assigned_to?.some(rel => String(rel.user_id) === String(userId))
      );

  // Admin stats
  const openCount = tickets.filter((t) => t.status === "OPEN").length;
  const unacknowledgedAlerts = alerts.filter(a => !a.tickets || a.tickets.length === 0).length;
  const activeTechsCount = technicians.length;

  // Tech workload count
  const assignedCount = ticketSource.length;

  // Machine status breakdown
  const critical = machines.filter((m) => m.health_status === "CRITICAL").length;
  const warning  = machines.filter((m) => m.health_status === "DEGRADING").length;
  const healthy  = machines.filter((m) => m.health_status === "HEALTHY").length;
  
  const criticalMachines = machines.filter((m) => m.health_status === "CRITICAL");
  // List filtering

  const displayFirstName = userName && userName.includes(" ") 
  ? userName.split(" ")[0] 
  : userName;

  return (
    <div className={styles.page}>
      {/* Welcome Header */}
      <div className={styles.header}>
        <div>
          <h2 className={styles.title}>
            Welcome back, {displayFirstName}
          </h2>
          {/* <p className={styles.sub}>Systems overview for your shift</p> */}
        </div>
      </div>

      {/* Numerical Insights */}
      <div className={styles.statStrip}>
        <div className={styles.statItem} style={{ borderTopColor: statColors.open }}>
          <span className={styles.statNum}>{openCount}</span>
          <span className={styles.statLbl}>Open Tickets</span>
        </div>

        {isAdmin ? (
          <>
            <div className={styles.statItem} style={{ borderTopColor: statColors.unacknowledged }}>
              <span className={styles.statNum}>{unacknowledgedAlerts}</span>
              <span className={styles.statLbl}>Unacknowledged Alerts</span>
            </div>
            <div className={styles.statItem} style={{ borderTopColor: statColors.activeTechs }}>
              <span className={styles.statNum}>{activeTechsCount}</span>
              <span className={styles.statLbl}>Total Technicians</span>
            </div>
          </>
        ) : (
          <div className={styles.statItem} style={{ borderTopColor: statColors.inprogress }}>
            <span className={styles.statNum}>{assignedCount}</span>
            <span className={styles.statLbl}>Assigned To Me</span>
          </div>
        )}

        <div className={styles.statItem} style={{ borderTopColor: statColors.critical }}>
          <span className={styles.statNum}>{critical}</span>
          <span className={styles.statLbl}>Critical Machines</span>
        </div>
        <div className={styles.statItem} style={{ borderTopColor: statColors.warning }}>
          <span className={styles.statNum}>{warning}</span>
          <span className={styles.statLbl}>Degrading Machines</span>
        </div>
        <div className={styles.statItem} style={{ borderTopColor: statColors.healthy }}>
          <span className={styles.statNum}>{healthy}</span>
          <span className={styles.statLbl}>Healthy Machines</span>
        </div>
      </div>

      {/* Data Visualization Grid */}
      <div className={styles.grid}>
        {/* Machine Alert List */}
        {/* Machine Alert List */}
  <div className={styles.card}>
    <h3 className={styles.cardTitle}>
      Critical machines
    </h3>
    <div className={styles.list}>
      {criticalMachines.length > 0 ? (
        criticalMachines.map((m) => (
          <div key={m.id || m.machine_id} className={styles.listRow}>
            <div className={styles.listLeft}>
              <span className={styles.listId}>{m.id || m.machine_id}</span>
              <span className={styles.listName}>{m.machine_type || "Unnamed Machine"}</span>
            </div>
            <div className={styles.listRight}>
              <span className={styles.listSub}>{m.location || "No Location"}</span>
              {/* <span className={`${styles.badge} ${styles.criticalBadge}`}>
                Critical
              </span> */}
            </div>
          </div>
        ))  
      ) : (
        <div className={styles.emptyList}>All machines operational. No critical alerts.</div>
      )}
    </div>
  </div>

        {/* Dynamic List: Techs or Active Tasks */}
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>
            {isAdmin ? "Registered Technicians" : "My Active Tickets"}
          </h3>
          <div className={styles.list}>
            {isAdmin ? (
              technicians.map((t) => (
                <div key={t.user_id || t.email} className={styles.listRow}>
                  <div className={styles.listLeft}>
                    <span className={styles.listName}>{t.name}</span>
                    <span className={styles.listSub}>{t.email}</span>
                  </div>
                  <div className={styles.listRight}>
                    <span className={`${styles.badge} ${styles.activeBadge}`}>
                      Active
                    </span>
                  </div>
                </div>
              ))
            ) : ticketSource.length > 0 ? (
              ticketSource.map((t) => (
                <div key={t.ticket_id} className={styles.listRow}>
                  <div className={styles.listLeft}>
                    <span className={styles.listId}>#TKT-{t.ticket_id}</span>
                    <span className={styles.listName}>{t.machine_id}</span>
                  </div>
                  <div className={styles.listRight}>
                    <span className={`${styles.badge} ${styles[t.priority.toLowerCase() + "Badge"]}`}>
                      {t.priority}
                    </span>
                    <span className={`${styles.badge} ${styles[t.status.toLowerCase().replace("-", "") + "Status"]}`}>
                      {t.status}
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <div className={styles.emptyList}>All caught up! No tickets assigned.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;