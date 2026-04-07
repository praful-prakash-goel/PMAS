import { useState, useEffect } from "react";
import styles from "../styles/technicians.module.css";

const BASE_URL = "http://localhost:8000";

const Badge = ({ status }) => {
  // availability_status is a boolean from backend, true = active
  const isActive = status === true || status === "active";
  const cls = isActive ? styles.activeBadge : styles.inactiveBadge;
  return (
    <span className={`${styles.badge} ${cls}`}>
      <span className={styles.bdot} />
      {isActive ? "Active" : "Inactive"}
    </span>
  );
};

const Technicians = () => {
  const token = localStorage.getItem("token");

  const [technicians,  setTechnicians]  = useState([]);
  // const [loading,      setLoading]      = useState(true);
  const [showModal,    setShowModal]    = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);

  // viewTarget = technician open in details popup
  const [viewTarget,   setViewTarget]   = useState(null);

  // removed org_name from form — backend takes it from the logged-in admin's org
  const [form, setForm] = useState({ name: "", email: "", password: "" });

  // fetch all technicians in this org on mount
  useEffect(() => {
    const fetchTechnicians = async () => {
      try {
        const res = await fetch(`${BASE_URL}/admin/technicians`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json();
        setTechnicians(data);
      } catch (err) {
        console.error("Fetch error:", err);
      } finally {
        // setLoading(false);
      }
    };
    if (token) fetchTechnicians();
  }, [token]);

  // active technicians first, then inactive
  const sortedTechnicians = [...technicians].sort((a, b) => {
    const aActive = a.technician?.availability_status ?? true;
    const bActive = b.technician?.availability_status ?? true;
    return bActive - aActive;
  });

  // POST /admin/technicians — org_name is inferred from admin on backend
  const handleRegister = async (e) => {
    e.preventDefault();
    
    // Minimal change: combine form state with the stored org_name
    const payload = { 
      ...form, 
      org_name: localStorage.getItem("userOrg") 
    };

    try {
      const res = await fetch(`${BASE_URL}/admin/technicians`, {
        method: "POST",
        headers: { 
          Authorization: `Bearer ${token}`, 
          "Content-Type": "application/json" 
        },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const updated = await fetch(`${BASE_URL}/admin/technicians`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        setTechnicians(await updated.json());
        setForm({ name: "", email: "", password: "" });
        setShowModal(false);
      } else {
        const err = await res.json();
        alert(err.detail?.[0]?.msg || "Registration failed.");
      }
    } catch {
      alert("Registration failed. Check connection.");
    }
  };

  // DELETE /admin/technicians/:email — backend uses email as identifier
  const handleRemove = async () => {
    try {
      const res = await fetch(`${BASE_URL}/admin/technicians/${deleteTarget.email}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        setTechnicians(technicians.filter((t) => t.email !== deleteTarget.email));
        setDeleteTarget(null);
      }
    } catch {
      alert("Delete failed.");
    }
  };

  // if (loading) return <div className={styles.loading}>Loading Technicians...</div>;

  return (
    <div className={styles.page}>

      <div className={styles.topBar}>
        <div>
          <div className={styles.pageTitle}>Technicians</div>
          <div className={styles.pageSubtitle}>{technicians.length} technicians registered</div>
        </div>
        <button className={styles.addBtn} onClick={() => setShowModal(true)}>
          + Register Technician
        </button>
      </div>

      <div className={styles.grid}>
        {sortedTechnicians.map((t, index) => (
        // Using a combination of user_id and index ensures React is satisfied 
        // even if the database returns a duplicate ID by mistake.
        <div key={t.user_id || index} className={`${styles.card} ...`}>
            <div className={styles.cardTop}>
              <div className={styles.avatar}>{t.name.split(" ").map(n => n[0]).join("")}</div>
              {/* availability_status is nested inside the technician relationship */}
              <Badge status={t.technician?.availability_status ?? true} />
            </div>
            <div className={styles.techName}>{t.name}</div>
            <div className={styles.techEmail}>{t.email}</div>
            <div className={styles.techOrg}>{t.org_name}</div>
            <div className={styles.cardFooter}>
              {/* loads this technician into viewTarget and opens the read-only popup */}
              <button className={styles.detailsBtn} onClick={() => setViewTarget(t)}>View Details →</button>
              <button className={styles.deleteBtn} onClick={() => setDeleteTarget(t)}>Remove</button>
            </div>
          </div>
        ))}
        {technicians.length === 0 && <p className={styles.empty}>No technicians registered yet.</p>}
      </div>

      {/* View Details Pop-Up — read only, no edit endpoint available for technicians */}
      {viewTarget && (
        <div className={styles.overlay} onClick={() => setViewTarget(null)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3>Technician Details</h3>
              <button className={styles.closeBtn} onClick={() => setViewTarget(null)}>✕</button>
            </div>
            <div className={styles.form}>
              <div className={styles.formRow}>
                <div className={styles.inputGroup}>
                  <label>Full Name</label>
                  <input type="text" value={viewTarget.name} disabled />
                </div>
                <div className={styles.inputGroup}>
                  <label>Email</label>
                  <input type="text" value={viewTarget.email} disabled />
                </div>
              </div>
              <div className={styles.formRow}>
                <div className={styles.inputGroup}>
                  <label>Organisation</label>
                  <input type="text" value={viewTarget.org_name} disabled />
                </div>
                <div className={styles.inputGroup}>
                  <label>Availability</label>
                  <input type="text" value={viewTarget.technician?.availability_status ? "Available" : "Unavailable"} disabled />
                </div>
              </div>
              <div className={styles.modalActions}>
                <button type="button" className={styles.cancelBtn} onClick={() => setViewTarget(null)}>Close</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Register Modal */}
      {showModal && (
        <div className={styles.overlay} onClick={() => setShowModal(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3>Register New Technician</h3>
              <button className={styles.closeBtn} onClick={() => setShowModal(false)}>✕</button>
            </div>
            <form onSubmit={handleRegister} className={styles.form}>
              <div className={styles.formRow}>
                <div className={styles.inputGroup}>
                  <label>Full Name</label>
                  <input type="text" placeholder="e.g. John Smith" value={form.name}
                    onChange={(e) => setForm({ ...form, name: e.target.value })} required />
                </div>
                <div className={styles.inputGroup}>
                  <label>Email</label>
                  <input type="email" placeholder="e.g. john@senseact.com" value={form.email}
                    onChange={(e) => setForm({ ...form, email: e.target.value })} required />
                </div>
              </div>
              {/* org_name removed — backend pulls it from the admin's token */}
              <div className={styles.inputGroup}>
                <label>Temporary Password</label>
                <input type="password" placeholder="Set a password" value={form.password}
                  onChange={(e) => setForm({ ...form, password: e.target.value })} required />
              </div>
              <div className={styles.modalActions}>
                <button type="button" className={styles.cancelBtn} onClick={() => setShowModal(false)}>Cancel</button>
                <button type="submit" className={styles.addBtn}>Register</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Delete Confirmation*/}
      {deleteTarget && (
        <div className={styles.overlay} onClick={() => setDeleteTarget(null)}>
          <div className={styles.confirmModal} onClick={(e) => e.stopPropagation()}>
            {/* <div className={styles.confirmIcon}>⚠️</div> */}
            <h3 className={styles.confirmTitle}>Remove Technician?</h3>
            <p className={styles.confirmText}>
              Are you sure you want to remove <strong>{deleteTarget.name}</strong> from the system? This action cannot be undone.
            </p>
            <div className={styles.confirmActions}>
              <button className={styles.cancelBtn} onClick={() => setDeleteTarget(null)}>Cancel</button>
              <button className={styles.confirmDeleteBtn} onClick={handleRemove}>Yes, Remove</button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};

export default Technicians;