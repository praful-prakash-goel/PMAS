import { useState, useEffect } from "react";
import monStyles from "../styles/monitoring.module.css";

const machines = [
  { id: "M-103", name: "CNC Mill",        location: "Production Line A" },
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

const thresholds = {
  temp:      { warning: 75, critical: 85  },
  vibration: { warning: 4,  critical: 6   },
  rpm:       { warning: 1500, critical: 1800 },
  load:      { warning: 80, critical: 90  },
};

// const daysSince = (dateStr) => {
//   const diff = Date.now() - new Date(dateStr).getTime();
//   return Math.floor(diff / (1000 * 60 * 60 * 24)) + " days";
// };

const getStatus = (key, value) => {
  if (value >= thresholds[key].critical) return "critical";
  if (value >= thresholds[key].warning)  return "warning";
  return "normal";
};

const Monitoring = () => {
  const [selectedId, setSelectedId] = useState("M-103");
  const [readings, setReadings] = useState(sampleReadings["M-103"]);

  // reset readings when machine changes
  const handleMachineChange = (e) => {
    const id = e.target.value;
    setSelectedId(id);
    setReadings(sampleReadings[id]);
  };

  // simulate live updates
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

  const selected = machines.find((m) => m.id === selectedId);

  const sensors = [
    { key: "temp",      label: "Temperature", value: readings.temp,      unit: "°C",   max: 120  },
    { key: "vibration", label: "Vibration",   value: readings.vibration, unit: "mm/s", max: 10   },
    { key: "rpm",       label: "RPM",         value: readings.rpm,       unit: "",     max: 2000 },
    { key: "load",      label: "Load",        value: readings.load,      unit: "%",    max: 100  },
  ];

  return (
    <div className={monStyles.page}>

      <div className={monStyles.topBar}>
        <div>
          <h2 className={monStyles.pageTitle}>Monitoring</h2>
          <p className={monStyles.subtitle}>
            Live sensor data — <span className={monStyles.liveDot} /> updates every 2 seconds
          </p>        
        </div>
        <select className={monStyles.select} value={selectedId} onChange={handleMachineChange}>
          {machines.map((m) => (
            <option key={m.id} value={m.id}>{m.id} — {m.name}</option>
          ))}
        </select>
      </div>

      <div className={monStyles.machineStrip}>
        <span className={monStyles.stripId}>{selected.id}</span>
        <span className={monStyles.stripName}>{selected.name}</span>
        <span className={monStyles.stripLocation}>{selected.location}</span>
        <span className={monStyles.stripRuntime}>Runtime: {readings.runtime}</span>
      </div>

      <div className={monStyles.sensorGrid}>
        {sensors.map((s) => {
          const status = getStatus(s.key, s.value);
          const pct = Math.min((s.value / s.max) * 100, 100);
          return (
            <div key={s.key} className={`${monStyles.sensorCard} ${monStyles[status + "Card"]}`}>
              <div className={monStyles.sensorTop}>
                <span className={monStyles.sensorLabel}>{s.label}</span>
                <span className={`${monStyles.sensorBadge} ${monStyles[status + "Badge"]}`}>
                  {status.charAt(0).toUpperCase() + status.slice(1)}
                </span>
              </div>
              <div className={monStyles.sensorValue}>
                {s.value}{s.unit}
              </div>
              <div className={monStyles.bar}>
                <div className={`${monStyles.barFill} ${monStyles[status + "Bar"]}`} style={{ width: `${pct}%` }} />
              </div>
            </div>
          );
        })}
      </div>

      <div className={monStyles.bottomGrid}>
        <div className={monStyles.card}>
          <h3 className={monStyles.cardTitle}>Failure Risk</h3>
          <div className={monStyles.riskValue}>{readings.risk}%</div>
          <div className={monStyles.bar} style={{ marginTop: "0.75rem" }}>
            <div
              className={`${monStyles.barFill} ${readings.risk >= 60 ? monStyles.criticalBar : readings.risk >= 30 ? monStyles.warningBar : monStyles.normalBar}`}
              style={{ width: `${readings.risk}%` }}
            />
          </div>
          <p className={monStyles.riskLabel}>
            {readings.risk >= 60 ? "High Risk — Immediate attention needed"
              : readings.risk >= 30 ? "Moderate Risk — Monitor closely"
              : "Low Risk — Machine operating normally"}
          </p>
        </div>

        <div className={monStyles.card}>
          <h3 className={monStyles.cardTitle}>Remaining Useful Life</h3>
          <div className={monStyles.rulValue}>{readings.rul} days</div>
          <p className={monStyles.rulLabel}>Next maintenance in <strong>{readings.nextMaint} days</strong></p>
        </div>
      </div>

    </div>
  );
};

export default Monitoring;