import { useState } from "react";
import styles from "../styles/profile.module.css";

const BASE_URL = "http://localhost:8000";

const Profile = () => {
  const token = localStorage.getItem("token");
  const storedRole = localStorage.getItem("userRole"); 
  
  const [prof, setProf] = useState({
    name: localStorage.getItem("userName") || "User",
    email: localStorage.getItem("userEmail") || "",
    orgName: localStorage.getItem("userOrg") || "Nexus",
    role: storedRole || "ADMIN",
  });

  const [form, setForm] = useState({ ...prof });
  const [editing, setEditing] = useState(false);
  const [showPassModal, setShowPassModal] = useState(false);
  const [showVerifyModal, setShowVerifyModal] = useState(false);
  
  const [verifyPass, setVerifyPass] = useState("");
  const [passData, setPassData] = useState({ old: "", new: "", confirm: "" });

  const handleVerifyAndEdit = async () => {
  try {
    const rolePath = prof.role.toLowerCase();
    // We send a request to a verify endpoint (or use your update-info with no changes)
    const res = await fetch(`${BASE_URL}/${rolePath}/update-info`, {
      method: "PATCH",
      headers: { 
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json"
      },
      // We send the password but don't change any data yet
      body: JSON.stringify({
        old_password: verifyPass 
      }),
    });

    if (res.ok) {
      setEditing(true);
      setShowVerifyModal(false);
      // Keep verifyPass for the final save, or clear it if backend doesn't need it twice
    } else {
      const err = await res.json();
      alert(err.detail || "Incorrect password. Access denied.");
      setVerifyPass("");
    }
  } catch {
    alert("System error during verification.");
  }
};

  const handleUpdate = async () => {
    try {
      const rolePath = prof.role.toLowerCase();
      const res = await fetch(`${BASE_URL}/${rolePath}/update-info`, {
        method: "PATCH",
        headers: { 
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          name: form.name,
          email: form.email,
          old_password: verifyPass 
        }),
      });

      if (res.ok) {
        setProf({ ...form });
        setEditing(false);
        setVerifyPass("");
        localStorage.setItem("userName", form.name);
        localStorage.setItem("userEmail", form.email);
        // alert("Profile updated!");
      } else {
        const err = await res.json();
        alert(err.detail || "Update failed.");
      }
    } catch {
      alert("Server error during update.");
    }
  };

  const handlePasswordChange = async () => {
    if (passData.new !== passData.confirm) return alert("New passwords do not match.");
    
    try {
      const rolePath = prof.role.toLowerCase();
      const res = await fetch(`${BASE_URL}/${rolePath}/update-info`, {
        method: "PATCH",
        headers: { 
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          old_password: passData.old,
          new_password: passData.new
        }),
      });

      if (res.ok) {
        alert("Password updated successfully!");
        setShowPassModal(false);
        setPassData({ old: "", new: "", confirm: "" });
      } else {
        const err = await res.json();
        alert(err.detail || "Current password incorrect.");
      }
    } catch {
      alert("Server error.");
    }
  };

  const initials = prof.name.split(" ").map((n) => n[0]).join("");

  return (
    <div className={styles.page}>
      <div className={styles.topBar}>
        <div>
          <h2 className={styles.pageTitle}>My Profile</h2>
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.heroSection}>
          <div className={styles.avatarLarge}>{initials}</div>
          <div className={styles.heroText}>
            <h3>{prof.name}</h3>
            <span>{prof.role}</span>
          </div>
        </div>

        <div className={styles.formGrid}>
          <div className={styles.inputGroup}>
            <label>Full Name</label>
            <input 
              type="text" 
              value={editing ? form.name : prof.name} 
              disabled={!editing}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
            />
          </div>
          <div className={styles.inputGroup}>
            <label>Email Address</label>
            <input 
              type="email" 
              value={editing ? form.email : prof.email} 
              disabled={!editing}
              onChange={(e) => setForm({ ...form, email: e.target.value })}
            />
          </div>
          <div className={styles.inputGroup}>
            <label>Organisation</label>
            <input type="text" value={prof.orgName} disabled className={styles.disabledInput} />
          </div>
        </div>

        <div className={styles.bottomActions}>
          {!editing ? (
            <div className={styles.buttonRow}>
              <button className={styles.editBtn} onClick={() => setShowVerifyModal(true)}>Edit Profile</button>
              <button className={styles.dangerBtn} onClick={() => setShowPassModal(true)}>Change Password</button>
            </div>
          ) : (
            <div className={styles.buttonRow}>
              <button className={styles.cancelBtn} onClick={() => { setEditing(false); setVerifyPass(""); }}>Cancel</button>
              <button className={styles.saveBtn} onClick={handleUpdate}>Save Changes</button>
            </div>
          )}
        </div>
      </div>

      {showVerifyModal && (
        <div className={styles.overlay}>
          <div className={styles.modal}>
            <div className={styles.modalHeader}>
              <h3>Security Check</h3>
              <button className={styles.closeBtn} onClick={() => setShowVerifyModal(false)}>✕</button>
            </div>
            <p className={styles.modalText}>Enter your password:</p>
            <div className={styles.inputGroup}>
              <input 
                type="password" 
                placeholder="Password" 
                value={verifyPass} 
                onChange={(e) => setVerifyPass(e.target.value)} 
              />
            </div>
            <div className={styles.modalActions}>
              <button className={styles.saveBtn} onClick={handleVerifyAndEdit}>Verify</button>
            </div>
          </div>
        </div>
      )}

      {showPassModal && (
        <div className={styles.overlay}>
          <div className={styles.modal}>
            <div className={styles.modalHeader}>
              <h3>Update Password</h3>
              <button className={styles.closeBtn} onClick={() => setShowPassModal(false)}>✕</button>
            </div>
            <div className={styles.form}>
              <div className={styles.inputGroup}>
                <label>Current Password</label>
                <input type="password" value={passData.old} onChange={(e) => setPassData({...passData, old: e.target.value})} />
              </div>
              <div className={styles.inputGroup}>
                <label>New Password</label>
                <input type="password" value={passData.new} onChange={(e) => setPassData({...passData, new: e.target.value})} />
              </div>
              <div className={styles.inputGroup}>
                <label>Confirm New Password</label>
                <input type="password" value={passData.confirm} onChange={(e) => setPassData({...passData, confirm: e.target.value})} />
              </div>
            </div>
            <div className={styles.modalActions}>
              <button className={styles.saveBtn} onClick={handlePasswordChange}>Change Password</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Profile;