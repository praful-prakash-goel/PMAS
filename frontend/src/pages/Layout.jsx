import { Outlet } from "react-router-dom";
import Navbar from "./Navbar";
import Sidebar from "./Sidebar";
import styles from "../styles/layout.module.css";

const Layout = () => (
  <div className={styles.app}>
    <Navbar />
    <div className={styles.body}>
      <Sidebar />
      <div className={styles.content}>
        <Outlet />
      </div>
    </div>
  </div>
);

export default Layout;