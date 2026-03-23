import { useState } from "react";
import styles from "../styles/technicians.module.css";

const sampleTechnicians = [
  { id: 1, name: "John Smith",    email: "john@senseact.com",  phone: "9876543210", assigned: 3, status: "active"   },
  { id: 2, name: "Sarah Johnson", email: "sarah@senseact.com", phone: "9123456780", assigned: 2, status: "active"   },
  { id: 3, name: "Mike Chen",     email: "mike@senseact.com",  phone: "9988776655", assigned: 1, status: "inactive" },
];

const Technicians = () => {
  const [technicians, setTechnicians] = useState(sampleTechnicians);
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState({ name: "", email: "", phone: "", password: "" });

  const handleRegister = async (e) => {
    e.preventDefault();

    // TODO: connect to backend
    // const response = await fetch("http://localhost:5000/register-technician", {
    //   method: "POST",
    //   headers: { "Content-Type": "application/json" },
    //   body: JSON.stringify(form),
    // });

    // optimistic UI update
    setTechnicians([...technicians, {
      id: technicians.length + 1,
      name: form.name,
      email: form.email,
      phone: form.phone,
      assigned: 0,
      status: "active",
    }]);

    setForm({ name: "", email: "", phone: "", password: "" });
    setShowModal(false);
  };

  return (
    <div className={styles.page}>

      <div className={styles.topBar}>
        <div>
          <h2 className={styles.pageTitle}>Technicians</h2>
          <p className={styles.subtitle}>{technicians.length} technicians registered</p>
        </div>
        <button className={styles.addBtn} onClick={() => setShowModal(true)}>
          + Register Technician
        </button>
      </div>

      <div className={styles.tableWrapper}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>#</th>
              <th>Name</th>
              <th>Email</th>
              <th>Phone</th>
              <th>Assigned Tickets</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {technicians.map((t, i) => (
              <tr key={t.id}>
                <td className={styles.index}>{i + 1}</td>
                <td className={styles.name}>{t.name}</td>
                <td>{t.email}</td>
                <td>{t.phone}</td>
                <td>{t.assigned}</td>
                <td>
                  <span className={`${styles.statusBadge} ${styles[t.status + "Status"]}`}>
                    {t.status.charAt(0).toUpperCase() + t.status.slice(1)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

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
                  <input
                    type="text"
                    placeholder="e.g. John Smith"
                    value={form.name}
                    onChange={(e) => setForm({ ...form, name: e.target.value })}
                    required
                  />
                </div>
                <div className={styles.inputGroup}>
                  <label>Email</label>
                  <input
                    type="email"
                    placeholder="e.g. john@senseact.com"
                    value={form.email}
                    onChange={(e) => setForm({ ...form, email: e.target.value })}
                    required
                  />
                </div>
              </div>
              <div className={styles.formRow}>
                <div className={styles.inputGroup}>
                  <label>Phone</label>
                  <input
                    type="tel"
                    placeholder="e.g. 9876543210"
                    value={form.phone}
                    onChange={(e) => setForm({ ...form, phone: e.target.value })}
                    required
                  />
                </div>
                <div className={styles.inputGroup}>
                  <label>Temporary Password</label>
                  <input
                    type="password"
                    placeholder="Set a password"
                    value={form.password}
                    onChange={(e) => setForm({ ...form, password: e.target.value })}
                    required
                  />
                </div>
              </div>
              <div className={styles.modalActions}>
                <button type="button" className={styles.cancelBtn} onClick={() => setShowModal(false)}>Cancel</button>
                <button type="submit" className={styles.addBtn}>Register</button>
              </div>
            </form>
          </div>
        </div>
      )}

    </div>
  );
};

export default Technicians;