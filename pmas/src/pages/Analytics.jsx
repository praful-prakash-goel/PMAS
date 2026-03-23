import anStyles from "../styles/analytics.module.css";

const machineHealth = [
  { label: "Healthy",  count: 18, color: "#52a876" },
  { label: "Warning",  count: 4,  color: "#e0a052" },
  { label: "Critical", count: 2,  color: "#e05252" },
];

const ticketStats = [
  { label: "Open",        count: 5,  color: "#e05252" },
  { label: "In Progress", count: 8,  color: "#e0a052" },
  { label: "Resolved",    count: 34, color: "#52a876" },
];

const technicianLoad = [
  { name: "John Smith",    assigned: 3, resolved: 12 },
  { name: "Sarah Johnson", assigned: 2, resolved: 9  },
  { name: "Mike Chen",     assigned: 1, resolved: 7  },
];

const recentActivity = [
  { ticket: "#TKT-1247", machine: "M-103", action: "Ticket opened",   by: "System",         time: "2 min ago"   },
  { ticket: "#TKT-1248", machine: "M-207", action: "Ticket assigned", by: "Admin",           time: "15 min ago"  },
  { ticket: "#TKT-1238", machine: "M-031", action: "Status updated",  by: "John Smith",     time: "1 hour ago"  },
  { ticket: "#TKT-1235", machine: "M-012", action: "Ticket resolved", by: "Mike Chen",      time: "1 day ago"   },
  { ticket: "#TKT-1230", machine: "M-089", action: "Ticket resolved", by: "Sarah Johnson",  time: "2 days ago"  },
];

const total = 24;

const Analytics = () => {
  return (
    <div className={anStyles.page}>

      <div className={anStyles.topBar}>
        <h2 className={anStyles.pageTitle}>Analytics</h2>
        <p className={anStyles.subtitle}>System-wide overview for administrators</p>
      </div>

      {/* Machine Health */}
      <section className={anStyles.section}>
        <h3 className={anStyles.sectionTitle}>Machine Health</h3>
        <div className={anStyles.statRow}>
          {machineHealth.map((item) => (
            <div key={item.label} className={anStyles.statCard} style={{ borderLeftColor: item.color }}>
              <div className={anStyles.statValue}>{item.count}</div>
              <div className={anStyles.statLabel}>{item.label}</div>
              <div className={anStyles.statBar}>
                <div
                  className={anStyles.statBarFill}
                  style={{ width: `${(item.count / total) * 100}%`, backgroundColor: item.color }}
                />
              </div>
              <div className={anStyles.statPct}>{Math.round((item.count / total) * 100)}% of {total}</div>
            </div>
          ))}
          <div className={anStyles.statCard} style={{ borderLeftColor: "#263A99" }}>
            <div className={anStyles.statValue}>{total}</div>
            <div className={anStyles.statLabel}>Total Machines</div>
          </div>
        </div>
      </section>

      {/* Ticket Stats */}
      <section className={anStyles.section}>
        <h3 className={anStyles.sectionTitle}>Ticket Overview</h3>
        <div className={anStyles.statRow}>
          {ticketStats.map((item) => (
            <div key={item.label} className={anStyles.statCard} style={{ borderLeftColor: item.color }}>
              <div className={anStyles.statValue}>{item.count}</div>
              <div className={anStyles.statLabel}>{item.label}</div>
              <div className={anStyles.statBar}>
                <div
                  className={anStyles.statBarFill}
                  style={{ width: `${(item.count / 47) * 100}%`, backgroundColor: item.color }}
                />
              </div>
            </div>
          ))}
          <div className={anStyles.statCard} style={{ borderLeftColor: "#263A99" }}>
            <div className={anStyles.statValue}>47</div>
            <div className={anStyles.statLabel}>Total Tickets</div>
          </div>
        </div>
      </section>

      <div className={anStyles.bottomGrid}>

        {/* Technician Workload */}
        <section className={anStyles.card}>
          <h3 className={anStyles.cardTitle}>Technician Workload</h3>
          <div className={anStyles.techList}>
            {technicianLoad.map((t) => (
              <div key={t.name} className={anStyles.techRow}>
                <div className={anStyles.techInfo}>
                  <span className={anStyles.techName}>{t.name}</span>
                  <span className={anStyles.techMeta}>{t.assigned} assigned · {t.resolved} resolved</span>
                </div>
                <div className={anStyles.techBars}>
                  <div className={anStyles.miniBarWrap}>
                    <div className={anStyles.miniBarFill} style={{ width: `${(t.assigned / 5) * 100}%`, backgroundColor: "#e0a052" }} />
                  </div>
                  <div className={anStyles.miniBarWrap}>
                    <div className={anStyles.miniBarFill} style={{ width: `${(t.resolved / 15) * 100}%`, backgroundColor: "#52a876" }} />
                  </div>
                </div>
              </div>
            ))}
            <div className={anStyles.legend}>
              <span style={{ color: "#e0a052" }}>■ Assigned</span>
              <span style={{ color: "#52a876" }}>■ Resolved</span>
            </div>
          </div>
        </section>

        {/* Recent Activity */}
        <section className={anStyles.card}>
          <h3 className={anStyles.cardTitle}>Recent Activity</h3>
          <div className={anStyles.activityList}>
            {recentActivity.map((a, i) => (
              <div key={i} className={anStyles.activityRow}>
                <div className={anStyles.activityLeft}>
                  <span className={anStyles.activityTicket}>{a.ticket}</span>
                  <span className={anStyles.activityAction}>{a.action} · {a.machine}</span>
                  <span className={anStyles.activityBy}>by {a.by}</span>
                </div>
                <span className={anStyles.activityTime}>{a.time}</span>
              </div>
            ))}
          </div>
        </section>

      </div>

    </div>
  );
};

export default Analytics;