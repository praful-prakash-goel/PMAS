import navbarStyles from "../styles/navbar.module.css";
import hamburgerButton from "../assets/hamburger_button.png";
import senseactLogo from "../assets/senseact_logo.png";
import notifBellIcon from "../assets/notif_bell_icon.png";
import { useNavigate } from "react-router-dom";

const Navbar = ({ toggleSidebar }) => {
  const navigate = useNavigate();

  return (
    <header className={navbarStyles.navbar}>
      
      {/* LEFT SECTION */}
      <div className={navbarStyles.leftSection}>
        <button className={navbarStyles.hamburgerBtn} onClick={toggleSidebar}>
          <img src={hamburgerButton} alt="Menu" />
        </button>

        <img
          src={senseactLogo}
          alt="Senseact Logo"
          className={navbarStyles.logo}
          onClick={() => navigate("/homepage")}
          style={{ cursor: "pointer" }}
        />
      </div>

      {/* RIGHT SECTION */}
      <div className={navbarStyles.rightSection}>
        <button className={navbarStyles.notificationBtn}>
          <img src={notifBellIcon} alt="Notifications" />
          <span className={navbarStyles.badge}>3</span>
        </button>
      </div>

    </header>
  );
};

export default Navbar;
