import { useState, useEffect } from "react";
import styles from "../styles/profile.module.css";

const BASE_URL = "http://localhost:8000";

const Profile = () => {
  const token = localStorage.getItem("token");
  
  // Initialize from storage to stay "accurate" to the logged-in user
  const [prof, setProf] = useState({
    Name: localStorage.getItem("userName") || "User",
    Email: localStorage.getItem("userEmail") || "",
    OrgName: localStorage.getItem("userOrg") || "",
    Role: localStorage.getItem("userRole") || "",
    userId: localStorage.getItem("userId"),
  });

  const [stats, setStats] = useState({ tickets: 0, machines: 0 });
  const [form, setForm] = useState({ ...prof });
  const [editing, setEditing] = useState(false);
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [passwords, setPasswords] = useState({ current: "", newPass: "", confirm: "" });

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch(`${BASE_URL}/users/stats/${prof.userId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
          const data = await res.json();
          setStats({ tickets: data.assigned_tickets, machines: data.managed_machines });
        }
      } catch{
        console.error("Stats fetch failed");
      }
    };
    if (token) fetchStats();
  }, [token, prof.userId]);

  const save = async () => {
    try {
      const res = await fetch(`${BASE_URL}/technician/update-info`, {
        method: "PATCH",
        headers: { 
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          name: form.Name,
          email: form.Email,
          old_password: passwords.current // Backend requires this for verification
        }),
      });

      if (res.ok) {
        setProf({ ...form });
        setEditing(false);
        localStorage.setItem("userName", form.Name);
        localStorage.setItem("userEmail", form.Email);
      }
    } catch{
      alert("Update failed.");
    }
  };

  const initials = prof.Name.split(" ").map((n) => n[0]).join("");

  return (
    <div className={styles.page}>
      <div className={styles.topBar}>
        <div>
          <div className={styles.pageTitle}>My Profile</div>
          <div className={styles.pageSubtitle}>Account Management</div>
        </div>
        <div className={styles.headerActions}>
          {!editing ? (
            <button className={styles.editBtn} onClick={() => setEditing(true)}>Edit Profile</button>
          ) : (
            <>
              <button className={styles.cancelBtn} onClick={() => setEditing(false)}>Cancel</button>
              <button className={styles.saveBtn} onClick={save}>Save Changes</button>
            </>
          )}
        </div>
      </div>

      <div className={styles.hero}>
        <div className={styles.avatar}>{initials}</div>
        <div className={styles.heroInfo}>
          <div className={styles.heroName}>{prof.Name}</div>
          <div className={styles.heroRole}>{prof.Role} · {prof.OrgName}</div>
          
          <div className={styles.heroStats}>
            <div className={styles.stat}>
              <div className={styles.statVal}>{stats.tickets}</div>
              <div className={styles.statLbl}>Tickets Assigned</div>
            </div>
            <div className={styles.stat}>
              <div className={styles.statVal}>{stats.machines}</div>
              <div className={styles.statLbl}>Machines Supervised</div>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.grid}>
        <div className={styles.card}>
          <div className={styles.sectionLbl}>Account Details</div>
          {!editing ? (
            <>
              <div className={styles.field}><div className={styles.fieldLbl}>Full Name</div><div className={styles.fieldVal}>{prof.Name}</div></div>
              <div className={styles.field}><div className={styles.fieldLbl}>Email</div><div className={styles.fieldVal}>{prof.Email}</div></div>
              <div className={styles.field}><div className={styles.fieldLbl}>Organisation</div><div className={styles.fieldVal}>{prof.OrgName}</div></div>
            </>
          ) : (
            <>
              <div className={styles.inputGroup}><label>Full Name</label><input type="text" value={form.Name} onChange={(e) => setForm({ ...form, Name: e.target.value })} /></div>
              <div className={styles.inputGroup}><label>Email Address</label><input type="email" value={form.Email} onChange={(e) => setForm({ ...form, Email: e.target.value })} /></div>
              <div className={styles.inputGroup}><label>Organisation (Fixed)</label><input type="text" value={prof.OrgName} disabled style={{ backgroundColor: '#f0f0f0' }} /></div>
            </>
          )}
          <button className={styles.dangerBtn} onClick={() => setShowPasswordModal(true)} style={{ marginTop: '20px' }}>Change Password</button>
        </div>
      </div>
    </div>
  );
};

export default Profile;