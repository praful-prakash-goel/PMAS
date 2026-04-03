import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import styles from "../styles/homepage.module.css";

// const CURRENT_USER = "John Smith";

// const allTickets = [
//   { id: "#TKT-1247", machine: "M-103", issue: "Replace cooling system",   priority: "critical", status: "open",        assignee: "John Smith"    },
//   { id: "#TKT-1240", machine: "M-156", issue: "Conveyor belt alignment",  priority: "low",      status: "open",        assignee: "John Smith"    },
//   { id: "#TKT-1238", machine: "M-031", issue: "Unusual noise from motor", priority: "high",     status: "in-progress", assignee: "John Smith"    },
//   { id: "#TKT-1248", machine: "M-207", issue: "Inspect vibration sensor", priority: "high",     status: "in-progress", assignee: "Sarah Johnson" },
//   { id: "#TKT-1245", machine: "M-045", issue: "Routine maintenance",      priority: "medium",   status: "in-progress", assignee: "Mike Chen"     },
//   { id: "#TKT-1235", machine: "M-012", issue: "Lubrication check",        priority: "low",      status: "resolved",    assignee: "John Smith"    },
// ];

// const machines = [
//   { id: "M-103", name: "CNC Mill",        location: "Production Line A", status: "critical" },
//   { id: "M-207", name: "Lathe",           location: "Warehouse B",       status: "critical" },
//   { id: "M-045", name: "Press",           location: "Assembly Floor",    status: "warning"  },
//   { id: "M-156", name: "Conveyor Belt",   location: "Production Line C", status: "warning"  },
//   { id: "M-012", name: "CNC Lathe",       location: "Production Line A", status: "healthy"  },
//   { id: "M-089", name: "Drill Press",     location: "Warehouse B",       status: "healthy"  },
//   { id: "M-031", name: "Milling Machine", location: "Production Line B", status: "healthy"  },
//   { id: "M-074", name: "Band Saw",        location: "Assembly Floor",    status: "healthy"  },
// ];

// const allTechnicians = [
//   { name: "John Smith",    email: "john@senseact.com",  assigned: 3, status: "active"   },
//   { name: "Sarah Johnson", email: "sarah@senseact.com", assigned: 2, status: "active"   },
//   { name: "Mike Chen",     email: "mike@senseact.com",  assigned: 1, status: "inactive" },
// ];

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
  const navigate = useNavigate();  

  //user_role no longer extracted from url parameters
  // const role = new URLSearchParams(location.search).get("user");
  // const isAdmin = role === "administrator";

  //user info from localStorage
  const role = localStorage.getItem("userRole");
  const userId = localStorage.getItem("userId");
  const userName = localStorage.getItem("userName"); 
  const token = localStorage.getItem("token");

  console.log(role);
  console.log(userName);

  //State for real data
  const [tickets, setTickets] = useState([]);
  const [machines, setMachines] = useState([]);
  const [technicians, setTechnicians] = useState([]);
  const [loading, setLoading] = useState(true);

  const isAdmin = role === "ADMIN";

  useEffect(() => {
    // Redirect if not logged in
    if (!token) {
      navigate("/");
      return;
    }

    const fetchData = async () => {
      try {
        const headers = { "Authorization": `Bearer ${token}` };

        // Fetch tickets and machines from your Python API
        const [tktRes, machRes] = await Promise.all([
          fetch(`http://localhost:8000/${role}/tickets`, { headers }),
          fetch(`http://localhost:8000/${role}/machines`, { headers })
        ]);

        // Error handling: If one fails, the whole dashboard should know
        if (!tktRes.ok || !machRes.ok) {
            throw new Error(`Fetch failed: ${tktRes.status} / ${machRes.status}`);
        }

        const tktData = await tktRes.json();
        const machData = await machRes.json();

        setTickets(tktData);
        setMachines(machData);
        
        if (isAdmin) {
          const techRes = await fetch("http://localhost:8000/admin/technicians", { headers });
          if (techRes.ok) setTechnicians(await techRes.json());
        }
      } catch (err) {
        console.error("Failed to load dashboard data", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [token, navigate, isAdmin]);

  if (loading) return <div className={styles.loading}>Loading Dashboard...</div>;

  // const ticketSource = isAdmin ? tickets : tickets.filter((t) => t.assignee_id === parseInt(userId));
  //LOGIC: Filter by User ID (Ensuring VARCHAR vs INT safety)
  const ticketSource = isAdmin 
    ? tickets 
    : tickets.filter((t) => String(t.assignee_id) === String(userId));

  // const ticketSource = isAdmin ? allTickets : allTickets.filter((t) => t.assignee === CURRENT_USER);
  const open       = ticketSource.filter((t) => t.status === "open").length;
  const inProgress = ticketSource.filter((t) => t.status === "in-progress").length;
  const resolved   = ticketSource.filter((t) => t.status === "resolved").length;

  const critical = machines.filter((m) => m.status === "critical").length;
  const warning  = machines.filter((m) => m.status === "warning").length;
  const healthy  = machines.filter((m) => m.status === "healthy").length;

  // Ensure this matches your backend JSON key (machine vs machine_id)
  const myMachineIds = [...new Set(ticketSource.map((t) => String(t.machine_id || t.machine)))];
  const myMachines = machines.filter((m) => myMachineIds.includes(String(m.id || m.machine_id)));
  const attentionMachines = machines.filter((m) => m.status !== "healthy");

  const displayFirstName = userName && userName.includes(" ") 
  ? userName.split(" ")[0] 
  : userName;

  return (
    <div className={styles.page}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <h2 className={styles.title}>
            Welcome back, {displayFirstName}
          </h2>
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
            {(isAdmin ? attentionMachines : myMachines).length > 0 ? (
              (isAdmin ? attentionMachines : myMachines).map((m) => (
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
              ))
            ) : (
              <div className={styles.emptyList}>No machines requiring immediate attention.</div>
            )}
          </div>
        </div>

        {/* Right: tickets or technicians */}
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>
            {isAdmin ? "Technicians" : "My Tickets"}
          </h3>
          <div className={styles.list}>
            {isAdmin ? (
              technicians.map((t) => (
                <div key={t.email} className={styles.listRow}>
                  <div className={styles.listLeft}>
                    <span className={styles.listName}>{t.name}</span>
                    <span className={styles.listSub}>{t.email}</span>
                  </div>
                  <div className={styles.listRight}>
                    <span className={styles.listSub}>{t.assigned_count} assigned</span>
                    <span className={`${styles.badge} ${t.status === "active" ? styles.activeBadge : styles.inactiveBadge}`}>
                      {t.status}
                    </span>
                  </div>
                </div>
              ))
            ) : ticketSource.length > 0 ? (
              ticketSource.map((t) => (
                <div key={t.id} className={styles.listRow}>
                  <div className={styles.listLeft}>
                    <span className={styles.listId}>{t.id}</span>
                    <span className={styles.listName}>{t.issue}</span>
                  </div>
                  <div className={styles.listRight}>
                    <span className={`${styles.badge} ${styles[t.priority + "Badge"]}`}>
                      {t.priority}
                    </span>
                    <span className={`${styles.badge} ${styles[t.status.replace("-", "") + "Status"]}`}>
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