import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Calculator, Wand2, Info, Download } from "lucide-react";

// ======= Utility helpers =======
const fmt = (n:number) => n.toLocaleString("en-IN", { maximumFractionDigits: 0 });
const clamp = (v:number, lo:number, hi:number) => Math.max(lo, Math.min(hi, v));

// India FY 2025-26 (AY 2026-27)
// Sources (summarized in chat):
// - New regime slabs per Union Budget 2025 (KPMG summary) 
// - 87A rebate up to ₹12,00,000 (effectively ₹12.75L for salaried after ₹75k std. deduction)
// - Standard deduction: ₹75,000 (new), ₹50,000 (old)
// - Cess 4%; Surcharge ladder; 37% not applicable under new regime (cap 25%)

// Types
type AgeBand = "<60" | "60-80" | ">80";

type Inputs = {
  resident: boolean;
  age: AgeBand;
  salaryIncome: number; // gross salary incl. allowances
  otherIncome: number; // interest, dividends, etc. (taxed at slab)

  // Special rate incomes
  stcg111A: number; // STCG @ 15% (listed equity where STT paid)
  ltcg112A: number; // LTCG @ 10% on equity over ₹1L
  otherLTCG112: number; // e.g., 20% with indexation (we'll let user choose rate)
  otherLTCG112Rate: number; // 10 or 20 typically

  // Old regime deductions
  old_std_ded: boolean; // apply standard deduction ₹50k automatically if salaried
  ded80C: number; // cap 1.5L
  ded80D: number; // medical insurance (self+family) generic cap 25k (we allow up to 75k as input, but cap below)
  ded80CCD1B: number; // NPS additional 50k
  housingLoanInterest: number; // self-occupied cap 2L (Loss from house property)
  otherDeductionsOld: number; // catch-all (subject to cap you know)

  // New regime allowances
  employerNPS_80CCD2: number; // allowed in new regime; user-provided amount
  familyPensionDeduction: number; // up to 25k (new regime)

  // Optimizer helpers
  investmentBudget: number; // how much can user additionally invest now
};

// Default inputs
const defaultInputs: Inputs = {
  resident: true,
  age: "<60",
  salaryIncome: 1800000,
  otherIncome: 60000,
  stcg111A: 0,
  ltcg112A: 0,
  otherLTCG112: 0,
  otherLTCG112Rate: 20,
  old_std_ded: true,
  ded80C: 100000,
  ded80D: 15000,
  ded80CCD1B: 0,
  housingLoanInterest: 0,
  otherDeductionsOld: 0,
  employerNPS_80CCD2: 0,
  familyPensionDeduction: 0,
  investmentBudget: 50000,
};

// Slabs — New Regime FY 2025-26 (per KPMG flash alert)
const newRegimeSlabs: { upto: number; rate: number }[] = [
  { upto: 400000, rate: 0 },
  { upto: 800000, rate: 5 },
  { upto: 1200000, rate: 10 },
  { upto: 1600000, rate: 15 },
  { upto: 2000000, rate: 20 },
  { upto: 2400000, rate: 25 },
  { upto: Infinity, rate: 30 },
];

// Slabs — Old Regime (unchanged)
const oldRegimeSlabsUnder60: { upto: number; rate: number }[] = [
  { upto: 250000, rate: 0 },
  { upto: 500000, rate: 5 },
  { upto: 1000000, rate: 20 },
  { upto: Infinity, rate: 30 },
];
const oldRegimeSlabs60to80: { upto: number; rate: number }[] = [
  { upto: 300000, rate: 0 },
  { upto: 500000, rate: 5 },
  { upto: 1000000, rate: 20 },
  { upto: Infinity, rate: 30 },
];
const oldRegimeSlabs80plus: { upto: number; rate: number }[] = [
  { upto: 500000, rate: 0 },
  { upto: 1000000, rate: 20 },
  { upto: Infinity, rate: 30 },
];

function computeSlabTax(income:number, slabs:{upto:number, rate:number}[]) {
  let remaining = income;
  let lower = 0;
  let tax = 0;
  let steps: {band:[number,number], amount:number, rate:number, tax:number}[] = [];
  for (const s of slabs) {
    const bandAmt = Math.max(0, Math.min(remaining, s.upto - lower));
    if (bandAmt > 0) {
      const t = bandAmt * s.rate / 100;
      tax += t;
      steps.push({ band: [lower+1, s.upto], amount: bandAmt, rate: s.rate, tax: t });
      remaining -= bandAmt;
      lower = s.upto;
    }
    if (remaining <= 0) break;
  }
  return { tax, steps };
}

// Surcharge helper
function computeSurcharge(baseTax:number, totalIncome:number, regime:"new"|"old") {
  // Thresholds
  const t = totalIncome;
  let rate = 0;
  if (t > 50000000 && regime === "old") rate = 37; // old regime only
  if (t > 20000000) rate = Math.max(rate, 25);
  if (t > 10000000) rate = Math.max(rate, 15);
  if (t > 5000000) rate = Math.max(rate, 10);
  if (regime === "new" && rate === 37) rate = 25; // safety cap
  const surcharge = baseTax * rate / 100;
  return { rate, surcharge };
}

function apply87ARebateNewRegime(baseTax:number, taxableExclSpecial:number) {
  // Rebate up to ₹60,000 if taxable income (excluding special rate income) <= ₹12,00,000
  if (taxableExclSpecial <= 1200000) {
    return Math.max(0, baseTax - 60000);
  }
  return baseTax;
}

function calcNewRegime(inputs: Inputs) {
  const stdDed = 75000; // standard deduction (salary/pension) new regime
  const familyPensionCap = 25000;

  const normalIncome = Math.max(0, inputs.salaryIncome + inputs.otherIncome - stdDed - clamp(inputs.familyPensionDeduction, 0, familyPensionCap) - inputs.employerNPS_80CCD2);

  // Slab tax on normal income
  const { tax: slabTax, steps } = computeSlabTax(normalIncome, newRegimeSlabs);

  // Special rate incomes
  const stcgTax = inputs.stcg111A * 0.15; // 15%
  const ltcgEqExemption = Math.max(0, inputs.ltcg112A - 100000);
  const ltcgEqTax = ltcgEqExemption * 0.10; // 10% over ₹1L
  const otherLTCGTax = inputs.otherLTCG112 * (inputs.otherLTCG112Rate / 100);

  // Apply 87A only to slab portion, excluding special incomes
  const slabTaxAfterRebate = apply87ARebateNewRegime(slabTax, normalIncome);
  let baseTax = slabTaxAfterRebate + stcgTax + ltcgEqTax + otherLTCGTax;

  // Surcharge
  const grossTotalIncome = inputs.salaryIncome + inputs.otherIncome + inputs.stcg111A + inputs.ltcg112A + inputs.otherLTCG112;
  const { rate: surchargeRate, surcharge } = computeSurcharge(baseTax, grossTotalIncome, "new");

  const cess = 0.04 * (baseTax + surcharge);
  const totalTax = Math.max(0, baseTax + surcharge + cess);

  return {
    regime: "new" as const,
    normalIncome, steps,
    slabTax, slabTaxAfterRebate,
    stcgTax, ltcgEqTax, otherLTCGTax,
    surchargeRate, surcharge,
    cess,
    totalTax
  };
}

function calcOldRegime(inputs: Inputs) {
  const stdDedOld = inputs.old_std_ded && inputs.salaryIncome > 0 ? 50000 : 0;
  const cap80C = clamp(inputs.ded80C, 0, 150000);
  const cap80D = clamp(inputs.ded80D, 0, 75000); // allow up to 75k overall (simplified)
  const capCCD1B = clamp(inputs.ded80CCD1B, 0, 50000);
  const hpLoss = -1 * clamp(inputs.housingLoanInterest, 0, 200000); // negative (loss from house property)
  const otherOld = Math.max(0, inputs.otherDeductionsOld);
  const gross = inputs.salaryIncome + inputs.otherIncome + inputs.stcg111A + inputs.ltcg112A + inputs.otherLTCG112;

  const normalGTI = Math.max(0, inputs.salaryIncome + inputs.otherIncome - stdDedOld + hpLoss);
  const deductions = cap80C + cap80D + capCCD1B + otherOld;
  const normalIncome = Math.max(0, normalGTI - deductions);

  // Old regime slabs depend on age
  const slabs = inputs.age === "<60" ? oldRegimeSlabsUnder60 : inputs.age === "60-80" ? oldRegimeSlabs60to80 : oldRegimeSlabs80plus;
  const { tax: slabTax, steps } = computeSlabTax(normalIncome, slabs);

  // Special rate incomes (same as new)
  const stcgTax = inputs.stcg111A * 0.15; // 15%
  const ltcgEqExemption = Math.max(0, inputs.ltcg112A - 100000);
  const ltcgEqTax = ltcgEqExemption * 0.10;
  const otherLTCGTax = inputs.otherLTCG112 * (inputs.otherLTCG112Rate / 100);

  // Section 87A rebate under old regime is only up to ₹5L total income (12,500), we'll apply simply here
  let slabTaxAfterRebate = slabTax;
  if (inputs.resident && normalIncome <= 500000 && inputs.stcg111A === 0 && inputs.ltcg112A === 0 && inputs.otherLTCG112 === 0) {
    slabTaxAfterRebate = Math.max(0, slabTax - 12500);
  }

  let baseTax = slabTaxAfterRebate + stcgTax + ltcgEqTax + otherLTCGTax;
  const grossTotalIncome = gross;
  const { rate: surchargeRate, surcharge } = computeSurcharge(baseTax, grossTotalIncome, "old");
  const cess = 0.04 * (baseTax + surcharge);
  const totalTax = Math.max(0, baseTax + surcharge + cess);

  return {
    regime: "old" as const,
    normalIncome, steps,
    slabTax, slabTaxAfterRebate,
    stcgTax, ltcgEqTax, otherLTCGTax,
    surchargeRate, surcharge,
    cess,
    totalTax,
    details: { stdDedOld, cap80C, cap80D, capCCD1B, hpLoss, otherOld, deductions }
  };
}

function suggestOptimization(inputs: Inputs) {
  // Simple heuristic: compute marginal tax rate under old regime on normal income, then recommend filling remaining 80C and 80CCD(1B) using budget.
  const oldRes = calcOldRegime(inputs);
  const remaining80C = Math.max(0, 150000 - clamp(inputs.ded80C, 0, 150000));
  const remainingCCD1B = Math.max(0, 50000 - clamp(inputs.ded80CCD1B, 0, 50000));
  const budget = inputs.investmentBudget;
  const to80C = Math.min(remaining80C, budget);
  const toCCD1B = Math.min(remainingCCD1B, budget - to80C);
  const totalAdd = to80C + toCCD1B;

  // Estimate marginal rate = last slab rate in old regime (excluding special incomes)
  const slabs = inputs.age === "<60" ? oldRegimeSlabsUnder60 : inputs.age === "60-80" ? oldRegimeSlabs60to80 : oldRegimeSlabs80plus;
  const lastRate = slabs.find(s => oldRes.normalIncome <= s.upto)?.rate ?? 30;
  const estSavings = totalAdd * lastRate / 100; // ignore cess/surcharge for quick estimate

  return {
    to80C, toCCD1B, totalAdd, estSavings, lastRate,
    note: "Investing the suggested amounts under the old regime could lower taxable income; compare regimes after applying."
  };
}

export default function TaxOptimizerApp() {
  const [inputs, setInputs] = useState<Inputs>(defaultInputs);
  const [showBreakdown, setShowBreakdown] = useState(true);

  const newRes = useMemo(() => calcNewRegime(inputs), [inputs]);
  const oldRes = useMemo(() => calcOldRegime(inputs), [inputs]);
  const better = newRes.totalTax <= oldRes.totalTax ? "new" : "old";
  const opt = useMemo(() => suggestOptimization(inputs), [inputs]);

  const update = (k: keyof Inputs, v: any) => setInputs(prev => ({ ...prev, [k]: v }));

  const exportJSON = () => {
    const blob = new Blob([JSON.stringify({ inputs, newRes, oldRes }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "tax-calculation-FY2025-26.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="max-w-6xl mx-auto p-6">
        <header className="flex items-center gap-3 mb-6">
          <Calculator className="w-8 h-8" />
          <h1 className="text-2xl font-semibold">Optimized Indian Income Tax Calculator — FY 2025–26 (AY 2026–27)</h1>
        </header>

        <div className="grid md:grid-cols-3 gap-4">
          {/* Inputs Card */}
          <motion.div layout className="md:col-span-2 bg-white rounded-2xl shadow p-4">
            <h2 className="text-lg font-semibold mb-3">Your Details & Income</h2>
            <div className="grid sm:grid-cols-2 gap-3">
              <label className="flex items-center gap-2">Resident?
                <input type="checkbox" checked={inputs.resident} onChange={e=>update('resident', e.target.checked)} className="ml-2"/>
              </label>
              <label className="flex items-center gap-2">Age band
                <select className="w-full border rounded-lg p-2" value={inputs.age} onChange={e=>update('age', e.target.value as AgeBand)}>
                  <option>{"<60"}</option>
                  <option>{"60-80"}</option>
                  <option>{">80"}</option>
                </select>
              </label>
              <Num label="Salary income (₹)" value={inputs.salaryIncome} onChange={v=>update('salaryIncome', v)} />
              <Num label="Other income (interest/dividends) (₹)" value={inputs.otherIncome} onChange={v=>update('otherIncome', v)} />

              <div className="sm:col-span-2 mt-2">
                <h3 className="font-medium">Special rate income</h3>
                <div className="grid sm:grid-cols-3 gap-3 mt-2">
                  <Num label="STCG @111A (15%) (₹)" value={inputs.stcg111A} onChange={v=>update('stcg111A', v)} />
                  <Num label="LTCG @112A (10% over ₹1L) (₹)" value={inputs.ltcg112A} onChange={v=>update('ltcg112A', v)} />
                  <div className="grid grid-cols-2 gap-2">
                    <Num label="Other LTCG (₹)" value={inputs.otherLTCG112} onChange={v=>update('otherLTCG112', v)} />
                    <Num label="Rate %" value={inputs.otherLTCG112Rate} onChange={v=>update('otherLTCG112Rate', v)} />
                  </div>
                </div>
              </div>
            </div>

            <div className="grid sm:grid-cols-2 gap-4 mt-4">
              <div className="bg-slate-50 rounded-xl p-3">
                <h3 className="font-semibold">Old Regime — Deductions</h3>
                <label className="flex items-center gap-2 mt-2">
                  <input type="checkbox" checked={inputs.old_std_ded} onChange={e=>update('old_std_ded', e.target.checked)} />
                  <span>Apply standard deduction (₹50,000)</span>
                </label>
                <Num label="Section 80C (cap ₹1.5L)" value={inputs.ded80C} onChange={v=>update('ded80C', v)} />
                <Num label="Section 80D — Medical (cap ~₹75k total)" value={inputs.ded80D} onChange={v=>update('ded80D', v)} />
                <Num label="NPS 80CCD(1B) (cap ₹50k)" value={inputs.ded80CCD1B} onChange={v=>update('ded80CCD1B', v)} />
                <Num label="Home loan interest (self-occupied, cap ₹2L)" value={inputs.housingLoanInterest} onChange={v=>update('housingLoanInterest', v)} />
                <Num label="Other deductions (old)" value={inputs.otherDeductionsOld} onChange={v=>update('otherDeductionsOld', v)} />
              </div>
              <div className="bg-slate-50 rounded-xl p-3">
                <h3 className="font-semibold">New Regime — Allowed</h3>
                <p className="text-sm text-slate-600">Includes standard deduction ₹75,000 automatically.</p>
                <Num label="Employer NPS 80CCD(2) (₹)" value={inputs.employerNPS_80CCD2} onChange={v=>update('employerNPS_80CCD2', v)} />
                <Num label="Family pension deduction (cap ₹25k)" value={inputs.familyPensionDeduction} onChange={v=>update('familyPensionDeduction', v)} />
              </div>
            </div>

            <div className="mt-4 bg-white rounded-xl">
              <h3 className="font-semibold mb-2">Optimization Helper</h3>
              <div className="grid sm:grid-cols-3 gap-3 items-end">
                <Num label="Budget to invest now (₹)" value={inputs.investmentBudget} onChange={v=>update('investmentBudget', v)} />
                <div className="sm:col-span-2 bg-emerald-50 rounded-xl p-3 border border-emerald-200">
                  <div className="flex items-center gap-2 mb-1"><Wand2 className="w-4 h-4"/><span className="font-medium">Suggestion (Old regime)</span></div>
                  <p className="text-sm">Invest ₹{fmt(opt.to80C)} in 80C and ₹{fmt(opt.toCCD1B)} in NPS 80CCD(1B). Estimated tax saving ₹{fmt(Math.round(opt.estSavings))} at marginal rate ~{opt.lastRate}% (excl. cess/surcharge).</p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Results Card */}
          <motion.div layout className="bg-white rounded-2xl shadow p-4">
            <h2 className="text-lg font-semibold mb-2">Results (Estimate)</h2>
            <div className="space-y-3">
              <ResultBlock title="New Regime" highlight={better === "new"} value={newRes.totalTax} breakdown={[
                ["Normal income (after SD, etc.)", newRes.normalIncome],
                ["Slab tax", newRes.slabTax],
                ["87A rebate (applied in slab)", newRes.slabTax - newRes.slabTaxAfterRebate],
                ["STCG @15%", newRes.stcgTax],
                ["LTCG Eq @10% (over ₹1L)", newRes.ltcgEqTax],
                ["Other LTCG", newRes.otherLTCGTax],
                ["Surcharge", newRes.surcharge],
                ["Cess @4%", newRes.cess],
              ]} />

              <ResultBlock title="Old Regime" highlight={better === "old"} value={oldRes.totalTax} breakdown={[
                ["Normal income (after deductions)", oldRes.normalIncome],
                ["Slab tax", oldRes.slabTax],
                ["87A rebate (if any)", oldRes.slabTax - oldRes.slabTaxAfterRebate],
                ["STCG @15%", oldRes.stcgTax],
                ["LTCG Eq @10% (over ₹1L)", oldRes.ltcgEqTax],
                ["Other LTCG", oldRes.otherLTCGTax],
                ["Surcharge", oldRes.surcharge],
                ["Cess @4%", oldRes.cess],
              ]} />

              <div className="p-3 rounded-xl border bg-slate-50">
                <p className="text-sm">✅ Recommended regime: <span className="font-semibold uppercase">{better}</span></p>
                <p className="text-sm">Tax difference: ₹{fmt(Math.abs(Math.round(newRes.totalTax - oldRes.totalTax)))} {newRes.totalTax === oldRes.totalTax ? '' : better==='new' ? 'saved vs old' : 'saved vs new'}</p>
              </div>

              <button onClick={exportJSON} className="mt-2 inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-slate-900 text-white hover:bg-slate-800">
                <Download className="w-4 h-4"/> Export JSON
              </button>

              <div className="text-xs text-slate-500 flex gap-2 mt-2">
                <Info className="w-4 h-4 mt-0.5"/>
                <p>
                  This tool estimates taxes using FY 2025–26 slabs (new regime) and common rules. It excludes many edge-cases (rebates, exemptions, losses, HRA specifics, surcharge marginal relief).
                  For official rules and filing, consult a CA or the Income Tax Department.
                </p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Optional Breakdown Table */}
        <div className="mt-6">
          <button onClick={()=>setShowBreakdown(!showBreakdown)} className="text-sm underline">{showBreakdown ? 'Hide' : 'Show'} slab breakdowns</button>
          {showBreakdown && (
            <div className="grid md:grid-cols-2 gap-4 mt-3">
              <BreakdownTable title="New Regime Slab Steps" steps={newRes.steps} />
              <BreakdownTable title="Old Regime Slab Steps" steps={oldRes.steps} />
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

function Num({label, value, onChange}:{label:string, value:number, onChange:(v:number)=>void}){
  return (
    <label className="text-sm">
      <div className="mb-1 text-slate-700">{label}</div>
      <input type="number" step="100" className="w-full border rounded-lg p-2" value={value} onChange={e=>onChange(Number(e.target.value || 0))} />
    </label>
  );
}

function ResultBlock({title, value, breakdown, highlight}:{title:string, value:number, breakdown:[string, number][], highlight:boolean}){
  return (
    <div className={`rounded-xl border p-3 ${highlight ? 'ring-2 ring-emerald-400' : ''}`}>
      <div className="flex items-baseline justify-between">
        <h3 className="font-semibold">{title}</h3>
        <div className="text-lg font-bold">₹{fmt(Math.round(value))}</div>
      </div>
      <ul className="mt-2 text-sm space-y-1">
        {breakdown.map(([k,v],i)=> (
          <li key={i} className="flex justify-between"><span className="text-slate-600">{k}</span><span>₹{fmt(Math.round(v))}</span></li>
        ))}
      </ul>
    </div>
  );
}

function BreakdownTable({title, steps}:{title:string, steps:{band:[number,number], amount:number, rate:number, tax:number}[]}){
  return (
    <div className="bg-white rounded-2xl shadow p-4">
      <h4 className="font-semibold mb-2">{title}</h4>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-600">
              <th className="py-2">Band (₹)</th>
              <th className="py-2">Amount</th>
              <th className="py-2">Rate</th>
              <th className="py-2">Tax</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((s, idx)=> (
              <tr key={idx} className="border-t">
                <td className="py-2">{fmt(s.band[0])}–{fmt(s.band[1])}</td>
                <td className="py-2">₹{fmt(Math.round(s.amount))}</td>
                <td className="py-2">{s.rate}%</td>
                <td className="py-2">₹{fmt(Math.round(s.tax))}</td>
              </tr>
            ))}
            {steps.length===0 && (
              <tr><td colSpan={4} className="py-4 text-center text-slate-500">No taxable slab income</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
