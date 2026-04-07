import { useState, useEffect } from "react";
import styles from "../styles/machines.module.css";

const BASE_URL = "http://localhost:8000";

// days since last service = how long the machine has been running without maintenance
const calculateRuntime = (lastService) => {
  if (!lastService) return "0 days";
  const diff = Date.now() - new Date(lastService).getTime();
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  return days < 0 ? "0 days" : `${days} days`;
};

// small colored badge showing machine health — normalizes status to uppercase just in case
const Badge = ({ status }) => {
  const s = status?.toUpperCase();
  const cls = {
    CRITICAL: styles.criticalBadge,
    DEGRADING:  styles.degradingBadge,
    HEALTHY:  styles.healthyBadge,
  };
  return (
    <span className={`${styles.badge} ${cls[s] || styles.healthyBadge}`}>
      <span className={styles.bdot} />
      {s || "UNKNOWN"}
    </span>
  );
};

const Machines = () => {
  // grab auth info from localStorage — set during login
  const role    = localStorage.getItem("userRole");
  const token   = localStorage.getItem("token");
  const isAdmin = role === "ADMIN";

  const [machines,       setMachines]       = useState([]);
  // const [loading,        setLoading]        = useState(true);
  const [locationFilter, setLocationFilter] = useState("all");
  const [statusFilter,   setStatusFilter]   = useState("all");

  // viewTarget = the machine currently open in the details/edit popup
  const [viewTarget,   setViewTarget]   = useState(null);
  const [isEditing,    setIsEditing]    = useState(false);

  // deleteTarget = machine staged for deletion, triggers the confirm modal
  const [deleteTarget,  setDeleteTarget]  = useState(null);
  const [showAddModal,  setShowAddModal]  = useState(false);
  const [addForm,       setAddForm]       = useState({
    machine_type: "", location: "", installation_date: "",
  });

  // fetch machines on mount — admin gets all, technician gets only assigned ones
  useEffect(() => {
    const fetchMachines = async () => {
      try {
        const prefix = isAdmin ? "admin" : "technician";
        const res = await fetch(`${BASE_URL}/${prefix}/machines`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json();
        setMachines(data);
      } catch (err) {
        console.error("Fetch error:", err);
      } finally {
        // setLoading(false);
      }
    };
    if (token) fetchMachines();
  }, [token, isAdmin]);

  // build location options dynamically from whatever the database returns
  const uniqueLocations = [...new Set(machines.map((m) => m.location))].filter(Boolean);

  const filtered = machines.filter((m) => {
    const locMatch    = locationFilter === "all" || m.location === locationFilter;
    const statusMatch = statusFilter   === "all" || m.health_status === statusFilter;
    return locMatch && statusMatch;
  });

  // POST /admin/machines — machine_id is auto-incremented on backend so we don't send it
  const handleAdd = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch(`${BASE_URL}/admin/machines`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify(addForm),
      });
      // console.log(addForm)
      if (res.ok) {
        // refetch to get the new machine with its auto-generated ID
        const updated = await fetch(`${BASE_URL}/admin/machines`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        setMachines(await updated.json());
        setAddForm({ machine_type: "", location: "", installation_date: "" });
        setShowAddModal(false);
      } else {
        const err = await res.json();
        alert(err.detail || "Add failed.");
      }
    } catch {
      alert("Add failed. Check your connection.");
    }
  };

  // PATCH /admin/machine/:id — only location, health_status and last_service_date are editable
  const handleUpdate = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch(`${BASE_URL}/admin/machine/${viewTarget.machine_id}`, {
        method: "PATCH",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          location:          viewTarget.location,
          health_status:     viewTarget.health_status,
          last_service_date: viewTarget.last_service_date,
        }),
      });
      // console.log(viewTarget.machine_id);
      // console.log(viewTarget.location);
      // console.log(viewTarget.health_status);
      // console.log(viewTarget.last_service_date);

      if (res.ok) {
        // update local state so UI reflects changes immediately without another fetch
        setMachines(machines.map((m) => m.machine_id === viewTarget.machine_id ? viewTarget : m));
        setIsEditing(false);
        setViewTarget(null);
      }
    } catch {
      alert("Update failed. Check your connection or permissions.");
    }
  };

  // DELETE /admin/machines/:id — only called after user confirms in the modal
  const handleDelete = async () => {
    try {
      const res = await fetch(`${BASE_URL}/admin/machines/${deleteTarget.machine_id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        setMachines(machines.filter((m) => m.machine_id !== deleteTarget.machine_id));
        setDeleteTarget(null);
      }
    } catch {
      alert("Delete failed.");
    }
  };

  // if (loading) return <div className={styles.loading}>Accessing Machine Registry...</div>;

  return (
    <div className={styles.page}>

      <div className={styles.topBar}>
        <div>
          <div className={styles.pageTitle}>Machines</div>
          <div className={styles.pageSubtitle}>{machines.length} total units discovered</div>
        </div>
        {/* only admins can add machines */}
        {isAdmin && (
          <button className={styles.addBtn} onClick={() => setShowAddModal(true)}>
            + Add Machine
          </button>
        )}
      </div>

      {/* filters — locations are pulled from DB, not hardcoded */}
      <div className={styles.filtersRow}>
        <div className={styles.selectWrap}>
          <select className={styles.select} value={locationFilter} onChange={(e) => setLocationFilter(e.target.value)}>
            <option value="all">All Locations</option>
            {uniqueLocations.map((loc) => <option key={loc} value={loc}>{loc}</option>)}
          </select>
          <span className={styles.selectArrow}>▾</span>
        </div>
        <div className={styles.selectWrap}>
          <select className={styles.select} value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="all">All Statuses</option>
            <option value="HEALTHY">HEALTHY</option>
            <option value="DEGRADING">DEGRADING</option>
            <option value="CRITICAL">CRITICAL</option>
          </select>
          <span className={styles.selectArrow}>▾</span>
        </div>
      </div>

      {/* Machine Cards Grid */}
      <div className={styles.grid}>
        {filtered.map((m) => (
          <div key={m.machine_id} className={styles.card}>
            <div className={styles.cardTop}>
              <span className={styles.machineId}>{m.machine_id}</span>
              <Badge status={m.health_status} />
            </div>
            <div className={styles.machineName}>{m.machine_type}</div>
            <div className={styles.machineLoc}>{m.location}</div>
            <div className={styles.runtime}>⏱ Runtime: {calculateRuntime(m.last_service_date)}</div>
            <div className={styles.cardFooter}>
              {/* clicking view details loads this machine into viewTarget and opens the popup */}
              <button className={styles.detailsBtn} onClick={() => { setViewTarget(m); setIsEditing(false); }}>
                View Details
              </button>
              {isAdmin && (
                <button className={styles.deleteBtn} onClick={() => setDeleteTarget(m)}>Delete</button>
              )}
            </div>
          </div>
        ))}
        {filtered.length === 0 && <p className={styles.empty}>No machines match the selected filters.</p>}
      </div>

      {/* View / Edit Pop-Up — same modal, switches between read-only and edit mode */}
      {viewTarget && (
        <div className={styles.overlay} onClick={() => { setViewTarget(null); setIsEditing(false); }}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3>{isEditing ? "Edit Machine Record" : "Technical Specifications"}</h3>
              <button className={styles.closeBtn} onClick={() => { setViewTarget(null); setIsEditing(false); }}>✕</button>
            </div>
            <form onSubmit={handleUpdate} className={styles.form}>
              {/* read-only fields — these never change after registration */}
              <div className={styles.formRow}>
                <div className={styles.inputGroup}>
                  <label>Machine ID</label>
                  <input type="text" value={viewTarget.machine_id} disabled />
                </div>
                <div className={styles.inputGroup}>
                  <label>Machine Type</label>
                  <input type="text" value={viewTarget.machine_type} disabled />
                </div>
              </div>
              <div className={styles.formRow}>
                <div className={styles.inputGroup}>
                  <label>Organization</label>
                  <input type="text" value={viewTarget.org_name} disabled />
                </div>
                <div className={styles.inputGroup}>
                  <label>Installation Date</label>
                  <input type="text" value={viewTarget.installation_date} disabled />
                </div>
              </div>
              <div className={styles.formRow}>
                <div className={styles.inputGroup}>
                  <label>Current Runtime</label>
                  <input type="text" value={calculateRuntime(viewTarget.last_service_date)} disabled />
                </div>
                {/* location is editable — machine might be moved */}
                <div className={styles.inputGroup}>
                  <label>Location</label>
                  <input type="text" value={viewTarget.location} disabled={!isEditing}
                    onChange={(e) => setViewTarget({ ...viewTarget, location: e.target.value })} />
                </div>
              </div>
              <div className={styles.formRow}>
                {/* health status and last service date are the main things updated after maintenance */}
                <div className={styles.inputGroup}>
                  <label>Health Status</label>
                  <select disabled={!isEditing} value={viewTarget.health_status}
                    onChange={(e) => setViewTarget({ ...viewTarget, health_status: e.target.value })}>
                    <option value="HEALTHY">HEALTHY</option>
                    <option value="DEGRADING">DEGRADING</option>
                    <option value="CRITICAL">CRITICAL</option>
                  </select>
                </div>
                <div className={styles.inputGroup}>
                  <label>Last Service Date</label>
                  <input type="date" value={viewTarget.last_service_date} disabled={!isEditing}
                    onChange={(e) => setViewTarget({ ...viewTarget, last_service_date: e.target.value })} />
                </div>
              </div>
              <div className={styles.modalActions}>
              {!isEditing ? (
                <>
                  <button 
                    type="button" 
                    className={styles.cancelBtn} 
                    onClick={(e) => {
                      e.preventDefault(); 
                      setViewTarget(null);
                    }}
                  >
                    Close
                  </button>
                  
                  {isAdmin && (
                    <button 
                      type="button" // Explicitly NOT a submit button
                      className={styles.addBtn} 
                      onClick={(e) => {
                        e.preventDefault();  // Stop form from thinking this is a submit
                        e.stopPropagation(); // Stop the modal overlay from closing
                        setIsEditing(true);  // JUST switch the UI to edit mode
                      }}
                    >
                      Modify Details
                    </button>
                  )}
                </>
              ) : (
                <>
                  <button 
                    type="button" 
                    className={styles.cancelBtn} 
                    onClick={() => setIsEditing(false)}
                  >
                    Cancel
                  </button>
                  {/* This is the ONLY button that should trigger handleUpdate */}
                  <button type="submit" className={styles.addBtn} >
                    Save to Database
                  </button>
                </>
              )}
            </div>
            </form>
          </div>
        </div>
      )}

      {/* Add Machine Modal — machine_id is auto-generated, so admin only fills type/location/date */}
      {showAddModal && (
        <div className={styles.overlay} onClick={() => setShowAddModal(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3>Add New Machine</h3>
              <button className={styles.closeBtn} onClick={() => setShowAddModal(false)}>✕</button>
            </div>
            <form onSubmit={handleAdd} className={styles.form}>
              <div className={styles.formRow}>
                <div className={styles.inputGroup}>
                  <label>Machine Type</label>
                  <input type="text" placeholder="e.g. CNC Mill"
                    value={addForm.machine_type}
                    onChange={(e) => setAddForm({ ...addForm, machine_type: e.target.value })} required />
                  {/* <select value={addForm.machine_type}
                    onChange={(e) => setAddForm({ ...addForm, machine_type: e.target.value })} required>
                    <option value="">Select Type</option>
                    <option value="CNC Machine">CNC Machine</option>
                    <option value="Lathe">Lathe</option>
                    <option value="Press">Press</option>
                    <option value="Conveyor">Conveyor</option>
                    <option value="Other">Other</option>
                  </select> */}
                </div>
                <div className={styles.inputGroup}>
                  <label>Location</label>
                  <input type="text" placeholder="e.g. Production Line A"
                    value={addForm.location}
                    onChange={(e) => setAddForm({ ...addForm, location: e.target.value })} required />
                </div>
              </div>
              <div className={styles.inputGroup}>
                <label>Installation Date</label>
                <input type="date" value={addForm.installation_date}
                  onChange={(e) => setAddForm({ ...addForm, installation_date: e.target.value })} required />
              </div>
              <div className={styles.modalActions}>
                <button type="button" className={styles.cancelBtn} onClick={() => setShowAddModal(false)}>Cancel</button>
                <button type="submit" className={styles.addBtn}>Add Machine</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal — shown when admin clicks Delete on a card */}
      {deleteTarget && (
        <div className={styles.overlay} onClick={() => setDeleteTarget(null)}>
          <div className={styles.confirmModal} onClick={(e) => e.stopPropagation()}>
            {/* <div className={styles.confirmIcon}>⚠️</div> */}
            <h3 className={styles.confirmTitle}>Delete Machine?</h3>
            <p className={styles.confirmText}>
              Are you sure you want to delete <strong>{deleteTarget.machine_id}</strong> ({deleteTarget.machine_type})? This cannot be undone.
            </p>
            <div className={styles.confirmActions}>
              <button className={styles.cancelBtn} onClick={() => setDeleteTarget(null)}>Cancel</button>
              <button className={styles.confirmDeleteBtn} onClick={handleDelete}>Yes, Delete</button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};

export default Machines;