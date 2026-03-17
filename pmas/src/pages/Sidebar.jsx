import styles from "../styles/sidebar.module.css";
import { useNavigate, useLocation } from "react-router-dom";

const Sidebar = ({ isOpen }) => {
  const navigate = useNavigate();
  const location = useLocation();

  // read role from URL query params
  const params = new URLSearchParams(location.search);
  const role = params.get("user");
  const isAdmin = role === "administrator";

  const navItems = [
    { label: "Overview",    path: "/homepage"    },
    { label: "Machines",    path: "/machines"    },
    { label: "Monitoring",  path: "/monitoring"  },
    { label: "Tickets",     path: "/tickets"     },
  ];

  // add Technicians tab only for admin
  if (isAdmin) {
    navItems.push({ label: "Technicians", path: "/technicians" });
    navItems.push({ label: "Analytics", path: "/analytics" });
  }

  return (
    <div className={`${styles.sidebar} ${isOpen ? styles.open : ""}`}>

      <div className={styles.navItems}>
        {navItems.map((item) => (
          <div
            key={item.path}
            className={`${styles.navItem} ${location.pathname === item.path ? styles.active : ""}`}
            onClick={() => navigate(`${item.path}${location.search}`)}
          >
            {item.label}
          </div>
        ))}
      </div>

      <div className={styles.footer}>
        <div>
          <div className={styles.userName}>User</div>
          <div className={styles.userRole}>
            {isAdmin ? "Administrator" : "Technician"}
          </div>
        </div>
        <button className={styles.logoutBtn} onClick={() => navigate("/login")}>
          Logout
        </button>
      </div>

    </div>
  );
};

export default Sidebar;