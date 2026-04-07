import { useState, useEffect } from "react";
import monStyles from "../styles/monitoring.module.css";

const BASE_URL = "http://localhost:8000";

/* // fields from class diagram and original sample data preserved as comments
const machines = [
  { id: "M-103", name: "CNC Mill",         location: "Production Line A" },
  { id: "M-207", name: "Lathe",           location: "Warehouse B"       },
  { id: "M-045", name: "Press",           location: "Assembly Floor"    },
  { id: "M-156", name: "Conveyor Belt",   location: "Production Line C" },
];

const sampleReadings = {
  "M-103": { temp: 87.5, vibration: 4.2, rpm: 1600, load: 82, runtime: "2,458 hrs", risk: 68, rul: 45,  nextMaint: 12 },
  "M-207": { temp: 72.1, vibration: 6.8, rpm: 1200, load: 91, runtime: "1,820 hrs", risk: 75, rul: 30,  nextMaint: 5  },
  "M-045": { temp: 61.3, vibration: 2.1, rpm: 950,  load: 55, runtime: "3,100 hrs", risk: 22, rul: 120, nextMaint: 45 },
  "M-156": { temp: 49.8, vibration: 1.4, rpm: 300,  load: 40, runtime: "980 hrs",   risk: 10, rul: 180, nextMaint: 90 },
};

const getStatus = (key, value) => {
  if (value >= thresholds[key].critical) return "critical";
  if (value >= thresholds[key].warning)  return "warning";
  return "normal";
};
*/

const Monitoring = () => {
  const [machineList, setMachineList] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [readings, setReadings] = useState(null);
  const [loading, setLoading] = useState(true);
  const token = localStorage.getItem("token");
  const role = localStorage.getItem("userRole");
  const rolePath = role.toLowerCase();

  // Fetch the actual machine list from the database
  useEffect(() => {
    const fetchMachines = async () => {
      try {
        const res = await fetch(`${BASE_URL}/${rolePath}/machines`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        const data = await res.json();
        setMachineList(data);
        if (data.length > 0) setSelectedId(data[0].machine_id);
      } catch {
        console.error("Failed to load machine list from DB");
      }
    };
    fetchMachines();
  }, [token]);

  // Poll the API every 5 seconds to sync with the CSV updates
  useEffect(() => {
    if (!selectedId) return;
    console.log(selectedId);

    const fetchMonitoringData = async () => {
      console.log(`${BASE_URL}/${rolePath}/monitoring/${selectedId}`)
      try {
        const res = await fetch(`${BASE_URL}/${rolePath}/monitoring/${selectedId}`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        const data = await res.json();
        setReadings(data);
        setLoading(false);
      } catch {
        console.error("Polling error: check if backend and CSVs are active");
      }
    };

    fetchMonitoringData();
    const interval = setInterval(fetchMonitoringData, 5000); 

    return () => clearInterval(interval);
  }, [selectedId, token]);

  /* // old simulation logic
  useEffect(() => {
    const base = sampleReadings[selectedId];
    const interval = setInterval(() => {
      setReadings({
        ...base,
        temp:      parseFloat((base.temp      + (Math.random() - 0.5) * 2).toFixed(1)),
        vibration: parseFloat((base.vibration + (Math.random() - 0.5) * 0.4).toFixed(1)),
        rpm:       Math.round(base.rpm        + (Math.random() - 0.5) * 40),
        load:      Math.round(base.load       + (Math.random() - 0.5) * 4),
      });
    }, 2000);
    return () => clearInterval(interval);
  }, [selectedId]);
  */

  if (loading || !readings) return <div className={monStyles.loading}>Syncing with sensors...</div>;

  const sensors = [
    { key: "temperature", label: "Temperature", value: readings.temperature, unit: "°C",  max: 600 },
    { key: "vibration",   label: "Vibration",   value: readings.vibration,   unit: "mm/s", max: 10  },
    { key: "rpm",         label: "RPM",         value: readings.rpm,         unit: "",     max: 2000 },
    { key: "torque",      label: "Torque",      value: readings.torque,      unit: "Nm",   max: 120 }, // Load replaced with Torque
  ];

  return (
    <div className={monStyles.page}>

      <div className={monStyles.topBar}>
        <div>
          <h2 className={monStyles.pageTitle}>Monitoring</h2>
          <p className={monStyles.subtitle}>
            Live sensor data — <span className={monStyles.liveDot} /> updates every 5 seconds
          </p>        
        </div>
        
        <select 
          className={monStyles.select} 
          value={selectedId} 
          onChange={(e) => setSelectedId(e.target.value)}
        >
          {machineList.map((m) => (
            <option key={m.machine_id} value={m.machine_id}>
              {m.machine_id} — {m.machine_type} ▾
            </option>
          ))}
        </select>
      </div>

      <div className={monStyles.machineStrip}>
        <span className={monStyles.stripId}>{readings.machine_id}</span>
        <span className={monStyles.stripName}>{readings.machine_type}</span>
        <span className={monStyles.stripLocation}>{readings.machine_location}</span>
        <span className={monStyles.stripRuntime}>
          Total Runtime: {readings.operating_hours?.toFixed(1)} hrs
        </span>
      </div>

      <div className={monStyles.sensorGrid}>
        {sensors.map((s) => {
          const pct = Math.min((s.value / s.max) * 100, 100);
          return (
            <div key={s.key} className={monStyles.sensorCard}>
              <div className={monStyles.sensorTop}>
                <span className={monStyles.sensorLabel}>{s.label}</span>
              </div>
              <div className={monStyles.sensorValue}>
                {s.value?.toFixed(1)}{s.unit}
              </div>
              <div className={monStyles.bar}>
                <div 
                  className={monStyles.barFill} 
                  style={{ width: `${pct}%`, backgroundColor: '#3b82f6' }} 
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className={monStyles.bottomGrid}>
        <div className={monStyles.card}>
          <h3 className={monStyles.cardTitle}>Time Since Last Maintenance</h3>
          {/* Convert hours from DB to Days for the display */}
          <div className={monStyles.riskValue}>
            {(readings.time_since_last_maint / 24).toFixed(1)} days
          </div>
          <div className={monStyles.bar} style={{ marginTop: "0.75rem", backgroundColor: '#e2e8f0' }}>
            <div className={monStyles.barFill} style={{ width: '100%', backgroundColor: '#64748b' }} />
          </div>
          <p className={monStyles.riskLabel}>Days of operation since the last maintenance check.</p>
        </div>

        <div className={monStyles.card}>
          <h3 className={monStyles.cardTitle}>Remaining Useful Life</h3>
          
          {/* RUL DAYS LOGIC */}
          <div className={monStyles.rulValue}>
            {(readings.rul_days || readings.rul) < 1 && (readings.rul_days || readings.rul) > 0 
              ? "< 1" 
              : (readings.rul_days || readings.rul || 0).toFixed(1)} 
            <span className={monStyles.unitLabel}> days</span>
          </div>

          <p className={monStyles.rulLabel}>
            Next maintenance scheduled in: 
            <strong> 
              {(readings.next_maintenance_days || readings.next_maint) < 1 && (readings.next_maintenance_days || readings.next_maint) > 0
                ? " < 1" 
                : ` ${(readings.next_maintenance_days || readings.next_maint || 0).toFixed(1)}`} 
              days
            </strong>
          </p>
        </div>
      </div>

    </div>
  );
};

export default Monitoring;