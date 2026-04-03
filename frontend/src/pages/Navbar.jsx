import styles from "../styles/navbar.module.css";
import { useLocation, useNavigate } from "react-router-dom";

const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <header className={styles.topbar}>

      <div className={styles.logo} onClick={() => navigate(`/homepage${location.search}`)}>
        SENSE<span>ACT</span>
      </div>

      {/* <div className={styles.right}>
        <div className={styles.livePill}>
          <span className={styles.liveDot} />
          Live
        </div>
        <div className={styles.notifWrap}>
          <span className={styles.notifBell}>🔔</span>
          <span className={styles.notifCount}>3</span>
        </div>
      </div> */}

    </header>
  );
};

export default Navbar;