# Supervisor Meeting Agenda
**Date:** [To be filled]  
**Attendees:** Diovandi Basheera Putra, Dr. Leonard P. Rusli  
**Duration:** 50-65 minutes (extended to include methodology briefing)  
**Objective:** Freeze technical scope, get approval on remaining work, and make critical decisions about 3D work and thesis structure

---

## Meeting Structure

### 1. Opening & Context (5 minutes)
- Brief recap of thesis progress since last meeting
- Current state: multi-geometry PINN + 3 geometries completed
- Objective: freeze scope and finalize remaining work

### 2. Methodology Briefing: PINN & MMC Explained (5-10 minutes)
**Important: This is the first in-depth discussion of the dual-approach methodology**

- **Problem Statement:** Discrete fastener placement optimization (where should screws go to maximize stiffness?)
- **Why Dual Approach:** Literature compares PINN vs FEA, MMC vs SIMP, but **no one has compared MMC vs PINN for discrete fasteners** - this thesis fills that gap
- **Approach A (PINN) Explained:**
  - Core idea: Differentiable FEA → training data → neural surrogate
  - "PINN turns expensive FEA into a fast differentiable black box. Upfront training cost, but then optimization is almost free."
  - Benefits: 10³× speedup after training, matches FEA within ≤1.1% across geometries
- **Approach B (MMC) Explained:**
  - Core idea: Represent screws as explicit geometric components, optimize directly via topology optimization
  - "MMC is a purist's method: classical mechanics, no learning. Transparent and precise, but you pay full FEA cost every iteration."
  - Benefits: No training needed, explicit manufacturable geometry
- **Why Compare:** Answers research questions about speed, quality, generalization, and implementation feasibility

### 3. Presentation: Current Results (10 minutes)
**Keep this concise - focus on key numbers and decisions needed**

- **Slide 1:** Problem + dual-approach summary
  - PINN vs MMC on discrete fasteners
  - Three geometries: L-bracket, tapered plate, ribbed channel
  
- **Slide 2:** Multi-geometry PINN performance
  - MAE ≤12.4 J, ≤1.1% MAPE on high-energy loads
  - ~10³× faster than FEA (0.009 ms vs 10.4 ms)
  - 1020 samples across 3 geometries / 5 loads
  
- **Slide 3:** MMC results
  - Convergence paths and final compliances
  - Three geometries completed
  - Runtime: 118 ms/iteration
  
- **Slide 4:** Comprehensive comparison
  - Quality, speed, training cost, generalization
  - Key trade-offs identified
  
- **Slide 5:** Remaining work
  - Ansys validation runs
  - MMC normalization to Joules
  - Dataset coverage figure update
  - Thesis writing + minor cleanup

### 4. Discussion Topics (20-25 minutes)

#### 3.1 Scope Freeze & Success Criteria
- [ ] Confirm current geometry set (L-bracket + tapered + ribbed) is sufficient
- [ ] Confirm Benchmark 2 (motor housing) and GA baseline are no longer required
- [ ] Confirm Ansys validation on 3-5 layouts with <20% error is sufficient

#### 3.2 3D Work Decision
- [ ] Is 3D benchmark required for passing?
- [ ] If required, is PINN-only 3D acceptable (MMC remains 2D-only)?
- [ ] Which 3D geometry preferred: extruded L-bracket or simplified motor-housing?
- [ ] Confirm 2D multi-geometry results are sufficient as core comparative study

#### 3.3 Grading Emphasis & Thesis Balance
- [ ] How should final thesis be balanced: comparative discussion vs. deep derivations vs. more experiments?
- [ ] Are MMC's normalized values on ribbed-channel acceptable if clearly explained, or must be fixed/renormalized first?

#### 3.4 Writing Expectations & Timeline
- [ ] Agree on chapter-delivery schedule (e.g., Ch.1-3 draft in 2 weeks, Ch.4-5 in 4 weeks, full draft in 6 weeks)
- [ ] How many feedback rounds is supervisor willing to do?
- [ ] What is the latest possible full-draft date before defense?

#### 3.5 Interpretation & Risk Items
- How actual work maps to original plan (pivot from "2 big 3D benchmarks + GA" to "3 well-controlled 2D/2.5D geometries")
- Interpretation of results: PINN amortized vs MMC per-iteration
- Risk items: Ansys not run, MMC normalization, ribbed-channel behavior

### 5. Decision Points (10 minutes)

**Critical decisions to make:**

1. **Thesis Scope Statement** (one sentence)
   - [ ] Approved scope: "Comparative study of multi-geometry PINN vs MMC on three 2D benchmark assemblies with Ansys validation on selected layouts"
   - [ ] Alternative scope (if 3D added): "Comparative study of multi-geometry PINN vs MMC on three 2D benchmark assemblies, with 3D PINN extension, and Ansys validation on selected layouts"

2. **Final Tasks Checklist** (sufficient to pass)
   - [ ] Run and document Ansys validation on N layouts (specify N: _____)
   - [ ] Normalize MMC compliances and regenerate unified comparison plots
   - [ ] Produce dataset coverage figure for the 1020-sample corpus
   - [ ] Deliver full thesis draft by date: _____
   - [ ] Additional tasks (if any): _____

3. **Stop Adding Experiments**
   - [ ] Confirm: stop adding new experiments, focus only on checklist items and writing
   - [ ] Any exceptions? (specify): _____

### 6. Action Items & Next Steps (5 minutes)

**Student actions:**
- [ ] Freeze codebase at current state
- [ ] Execute Ansys validation per agreed scope
- [ ] Normalize MMC compliances
- [ ] Begin thesis writing per agreed schedule
- [ ] Send meeting summary within 24 hours

**Supervisor actions:**
- [ ] Review and approve scope statement
- [ ] Confirm review schedule and availability
- [ ] Provide feedback on 3D decision

### 7. Closing (2 minutes)
- Confirm next meeting date: _____
- Confirm communication channel for questions
- Thank supervisor for time

---

## Notes Section

**Supervisor feedback/decisions:**
- 
- 
- 

**Open questions to follow up:**
- 
- 
- 

---

## Supporting Documents

- Executive Summary: `docs/meeting_materials/executive_summary.md`
- Presentation Slides: `docs/meeting_materials/presentation_slides.md`
- Decision Checklist: `docs/meeting_materials/decision_checklist.md`
- Comprehensive Results: `data/results/comprehensive_results_table.md`
- Figure Inventory: `data/results/figure_inventory.md`

