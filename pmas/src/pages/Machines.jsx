import { useState } from "react";
import { useLocation } from "react-router-dom";
import machineStyles from "../styles/machines.module.css";

const sampleMachines = [
  { id: "M-103", name: "CNC Mill",        type: "CNC Machine", location: "Production Line A", status: "critical", runtime: "2,458 hrs" },
  { id: "M-207", name: "Lathe",           type: "Lathe",       location: "Warehouse B",       status: "critical", runtime: "1,820 hrs" },
  { id: "M-045", name: "Press",           type: "Press",       location: "Assembly Floor",    status: "warning",  runtime: "3,100 hrs" },
  { id: "M-156", name: "Conveyor Belt",   type: "Conveyor",    location: "Production Line C", status: "warning",  runtime: "980 hrs"   },
  { id: "M-012", name: "CNC Lathe",       type: "Lathe",       location: "Production Line A", status: "healthy",  runtime: "512 hrs"   },
  { id: "M-089", name: "Drill Press",     type: "Press",       location: "Warehouse B",       status: "healthy",  runtime: "4,200 hrs" },
  { id: "M-031", name: "Milling Machine", type: "CNC Machine", location: "Production Line B", status: "healthy",  runtime: "1,340 hrs" },
  { id: "M-074", name: "Band Saw",        type: "Other",       location: "Assembly Floor",    status: "healthy",  runtime: "670 hrs"   },
];

const Machines = () => {
  const location = useLocation();
  const role = new URLSearchParams(location.search).get("user");
  const isAdmin = role === "administrator";

  const [machines, setMachines] = useState(sampleMachines);
  const [locationFilter, setLocationFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState({
    machineId: "", machineName: "", type: "", location: "", installDate: "", interval: 30,
  });

  const filtered = machines.filter((m) => {
    const locMatch = locationFilter === "all" || m.location.toLowerCase().includes(locationFilter);
    const statusMatch = statusFilter === "all" || m.status === statusFilter;
    return locMatch && statusMatch;
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    // TODO: connect to backend
    setMachines([...machines, {
      id: form.machineId,
      name: form.machineName,
      type: form.type,
      location: form.location,
      status: "healthy",
      runtime: "0 hrs",
    }]);
    setForm({ machineId: "", machineName: "", type: "", location: "", installDate: "", interval: 30 });
    setShowModal(false);
  };

  return (
    <div className={machineStyles.page}>

      <div className={machineStyles.topBar}>
        <h2 className={machineStyles.pageTitle}>Machines</h2>
        <div className={machineStyles.actions}>
          <select className={machineStyles.select} value={locationFilter} onChange={(e) => setLocationFilter(e.target.value)}>
            <option value="all">All Locations</option>
            <option value="production line a">Production Line A</option>
            <option value="production line b">Production Line B</option>
            <option value="production line c">Production Line C</option>
            <option value="warehouse">Warehouse</option>
            <option value="assembly">Assembly Floor</option>
          </select>
          <select className={machineStyles.select} value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="all">All Status</option>
            <option value="healthy">Healthy</option>
            <option value="warning">Warning</option>
            <option value="critical">Critical</option>
          </select>
          {isAdmin && (
            <button className={machineStyles.addBtn} onClick={() => setShowModal(true)}>
              + Add Machine
            </button>
          )}
        </div>
      </div>

      <div className={machineStyles.grid}>
        {filtered.map((machine) => (
          <div key={machine.id} className={`${machineStyles.card} ${machineStyles[machine.status]}`}>
            <div className={machineStyles.cardTop}>
              <span className={machineStyles.machineId}>{machine.id}</span>
              <span className={`${machineStyles.badge} ${machineStyles[machine.status + "Badge"]}`}>
                {machine.status.charAt(0).toUpperCase() + machine.status.slice(1)}
              </span>
            </div>
            <div className={machineStyles.machineName}>{machine.name}</div>
            <div className={machineStyles.meta}>
              <span>{machine.type}</span>
              <span>{machine.location}</span>
            </div>
            <div className={machineStyles.runtime}>Runtime: {machine.runtime}</div>
            <button className={machineStyles.detailsBtn}>View Details</button>
          </div>
        ))}
        {filtered.length === 0 && (
          <p className={machineStyles.empty}>No machines match the selected filters.</p>
        )}
      </div>

      {isAdmin && showModal && (
        <div className={machineStyles.overlay} onClick={() => setShowModal(false)}>
          <div className={machineStyles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={machineStyles.modalHeader}>
              <h3>Add New Machine</h3>
              <button className={machineStyles.closeBtn} onClick={() => setShowModal(false)}>✕</button>
            </div>
            <form onSubmit={handleSubmit} className={machineStyles.form}>
              <div className={machineStyles.formRow}>
                <div className={machineStyles.inputGroup}>
                  <label>Machine ID</label>
                  <input type="text" placeholder="M-XXX" value={form.machineId}
                    onChange={(e) => setForm({ ...form, machineId: e.target.value })} required />
                </div>
                <div className={machineStyles.inputGroup}>
                  <label>Machine Name</label>
                  <input type="text" placeholder="e.g. CNC Mill" value={form.machineName}
                    onChange={(e) => setForm({ ...form, machineName: e.target.value })} required />
                </div>
              </div>
              <div className={machineStyles.formRow}>
                <div className={machineStyles.inputGroup}>
                  <label>Type</label>
                  <select value={form.type} onChange={(e) => setForm({ ...form, type: e.target.value })} required>
                    <option value="">Select Type</option>
                    <option value="CNC Machine">CNC Machine</option>
                    <option value="Lathe">Lathe</option>
                    <option value="Press">Press</option>
                    <option value="Conveyor">Conveyor</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
                <div className={machineStyles.inputGroup}>
                  <label>Location</label>
                  <input type="text" placeholder="e.g. Production Line A" value={form.location}
                    onChange={(e) => setForm({ ...form, location: e.target.value })} required />
                </div>
              </div>
              <div className={machineStyles.formRow}>
                <div className={machineStyles.inputGroup}>
                  <label>Installation Date</label>
                  <input type="date" value={form.installDate}
                    onChange={(e) => setForm({ ...form, installDate: e.target.value })} required />
                </div>
                <div className={machineStyles.inputGroup}>
                  <label>Maintenance Interval (days)</label>
                  <input type="number" min="1" value={form.interval}
                    onChange={(e) => setForm({ ...form, interval: e.target.value })} required />
                </div>
              </div>
              <div className={machineStyles.modalActions}>
                <button type="button" className={machineStyles.cancelBtn} onClick={() => setShowModal(false)}>Cancel</button>
                <button type="submit" className={machineStyles.addBtn}>Add Machine</button>
              </div>
            </form>
          </div>
        </div>
      )}

    </div>
  );
};

export default Machines;