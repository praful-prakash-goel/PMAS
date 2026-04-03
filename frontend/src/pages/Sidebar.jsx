import styles from "../styles/sidebar.module.css";
import { useNavigate, useLocation } from "react-router-dom";

const Sidebar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // const params = new URLSearchParams(location.search);
  // const role = params.get("user");
  // const isAdmin = role === "administrator";
  // const page = location.pathname.replace("/", "") || "homepage";
  
  // real data from storage instead of the URL
  const role = localStorage.getItem("userRole");
  const userName = localStorage.getItem("userName");
  const isAdmin = role === "ADMIN";

  const path = location.pathname.replace("/", "");
  const currentPage = path === "" ? "homepage" : path;

  const navItems = [
    { key: "homepage",   label: "Dashboard"  },
    { key: "machines",   label: "Machines"   },
    { key: "tickets",    label: "Tickets"    },
    { key: "monitoring", label: "Monitoring" },
  ];

  // admin-only items
  if (isAdmin) {
    // navItems.push({ key: "analytics",   label: "Analytics"   });
    navItems.push({ key: "alerts", label: "Alerts" });
    navItems.push({ key: "technicians", label: "Technicians" });
  }

  const goTo = (key) => navigate(`/${key}`);

  //Logout handler
  const handleLogout = () => {
    localStorage.clear();
    navigate("/");
  };

  return (
    <aside className={styles.sidebar}>

      <nav className={styles.nav}>
        {navItems.map((item) => (
          <div
            key={item.key}
            className={`${styles.navItem} ${currentPage === item.key ? styles.active : ""}`}
            onClick={() => goTo(item.key)}
          >
            {item.label}
          </div>
        ))}
      </nav>

    <div className={styles.footer}>
      <div
        className={styles.userChip}
        onClick={() => navigate(`/profile`)}
      >
        <div className={styles.avatar}>{isAdmin ? "A" : "T"}</div>
        <div className={styles.userInfo}>
          <div className={styles.userName}>{userName}</div>
          <div className={styles.userRole}>{isAdmin ? "Admin" : "Technician"}</div>
        </div>
      </div>
        <button onClick={handleLogout} className={styles.logoutBtn}>
          Logout
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;