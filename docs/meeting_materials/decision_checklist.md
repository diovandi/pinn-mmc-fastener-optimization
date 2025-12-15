# Decision Checklist: Supervisor Meeting

**Purpose:** Capture all critical decisions during the meeting to avoid ambiguity later  
**Instructions:** Check boxes and fill in answers as decisions are made

---

## 1. Scope Freeze & Success Criteria

### 1.1 Geometry Set
- [ ] **Current geometry set is sufficient for thesis**
  - L-bracket (2 load cases)
  - Tapered plate (1 load case)
  - Ribbed channel (2 load cases)
  - **Decision:** ☐ Yes ☐ No
  - **Notes:** _____

- [ ] **Benchmark 2 (motor housing) is no longer required**
  - **Decision:** ☐ Yes ☐ No
  - **Notes:** _____

- [ ] **GA baseline is no longer required**
  - **Decision:** ☐ Yes ☐ No
  - **Notes:** _____

### 1.2 Ansys Validation Requirements
- [ ] **Number of layouts to validate:**
  - **Decision:** _____ layouts (suggested: 3-5)
  - **Which layouts:**
    - [ ] Best diff-FEA layout (L-bracket)
    - [ ] Best PINN layout (L-bracket)
    - [ ] Best MMC layout (L-bracket, each load)
    - [ ] At least one non-L-bracket geometry (tapered or ribbed)
    - [ ] Other: _____

- [ ] **Error threshold acceptable:**
  - **Decision:** <_____% error (suggested: <20%)
  - **Notes:** _____

- [ ] **Validation is sufficient if error < threshold:**
  - **Decision:** ☐ Yes ☐ No
  - **Notes:** _____

---

## 2. 3D Work Decision

### 2.1 3D Requirement
- [ ] **Is 3D benchmark required for passing?**
  - **Decision:** ☐ Required ☐ Optional ☐ Not required
  - **Notes:** _____

### 2.2 If 3D Required
- [ ] **Is PINN-only 3D acceptable?**
  - (MMC remains 2D-only, as 3D MMC is out of scope)
  - **Decision:** ☐ Yes ☐ No
  - **Notes:** _____

- [ ] **Which 3D geometry preferred?**
  - **Decision:** ☐ Extruded L-bracket ☐ Simplified motor-housing ☐ Other: _____
  - **Notes:** _____

- [ ] **Dataset size for 3D:**
  - **Decision:** _____ samples (suggested: 50-100)
  - **Notes:** _____

- [ ] **Timeline for 3D work:**
  - **Decision:** _____ weeks allocated
  - **Notes:** _____

### 2.3 2D Work Status
- [ ] **Current 2D multi-geometry results are sufficient as core comparative study**
  - (Regardless of 3D decision)
  - **Decision:** ☐ Yes ☐ No
  - **Notes:** _____

---

## 3. Grading Emphasis & Thesis Balance

### 3.1 Thesis Emphasis
- [ ] **How should final thesis be balanced?**
  - **Decision:** 
    - [ ] More weight on comparative discussion and generalization (current strong numbers)
    - [ ] More weight on deep derivations
    - [ ] More weight on additional experiments
    - [ ] Balanced across all three
  - **Notes:** _____

### 3.2 MMC Normalization
- [ ] **MMC normalized values on ribbed-channel:**
  - **Decision:** 
    - [ ] Acceptable if clearly explained (scaling and trends)
    - [ ] Must be fixed/renormalized before thesis submission
  - **Notes:** _____

- [ ] **MMC normalization to Joules:**
  - **Timeline:** 
    - [ ] Must be done before writing begins
    - [ ] Can be done during writing phase
    - [ ] Can be explained in text without conversion
  - **Notes:** _____

---

## 4. Writing Expectations & Timeline

### 4.1 Chapter Delivery Schedule
- [ ] **Chapters 1-3 draft (Introduction, Literature Review, Theoretical Framework):**
  - **Target date:** _____ weeks from meeting
  - **Decision:** ☐ Approved ☐ Needs adjustment
  - **Notes:** _____

- [ ] **Chapters 4-5 draft (Methodology, Results):**
  - **Target date:** _____ weeks from meeting
  - **Decision:** ☐ Approved ☐ Needs adjustment
  - **Notes:** _____

- [ ] **Full draft (all chapters):**
  - **Target date:** _____ weeks from meeting
  - **Latest possible date:** _____ (back-solve from defense date)
  - **Decision:** ☐ Approved ☐ Needs adjustment
  - **Notes:** _____

### 4.2 Review Process
- [ ] **Number of feedback rounds supervisor is willing to do:**
  - **Decision:** _____ rounds (suggested: 2-3)
  - **Notes:** _____

- [ ] **Review turnaround time:**
  - **Decision:** _____ weeks per round
  - **Notes:** _____

- [ ] **Communication method:**
  - **Decision:** ☐ Email ☐ In-person ☐ Virtual meeting ☐ Other: _____
  - **Notes:** _____

---

## 5. Final Thesis Scope Statement

### 5.1 Approved Scope
- [ ] **One-sentence thesis scope statement:**
  - **Decision:** 
    ```
    [Fill in approved statement]
    ```
  - **Notes:** _____

**Example options:**

- "Comparative study of multi-geometry PINN vs MMC on three 2D benchmark assemblies with Ansys validation on selected layouts"
- "Comparative study of multi-geometry PINN vs MMC on three 2D benchmark assemblies, with 3D PINN extension, and Ansys validation on selected layouts"

---

## 6. Final Tasks Checklist (Sufficient to Pass)

### 6.1 Technical Tasks
- [ ] **Run and document Ansys validation on N layouts**
  - N = _____ (from section 1.2)
  - **Timeline:** _____ weeks
  - **Decision:** ☐ Approved ☐ Needs adjustment

- [ ] **Normalize MMC compliances and regenerate unified comparison plots**
  - **Timeline:** _____ weeks
  - **Decision:** ☐ Approved ☐ Needs adjustment

- [ ] **Produce dataset coverage figure for 1020-sample corpus**
  - **Timeline:** _____ weeks
  - **Decision:** ☐ Approved ☐ Needs adjustment

- [ ] **Additional technical tasks (if any):**
  - _____
  - **Timeline:** _____ weeks
  - **Decision:** ☐ Approved ☐ Needs adjustment

### 6.2 Writing Tasks
- [ ] **Deliver full thesis draft by date:**
  - **Date:** _____
  - **Decision:** ☐ Approved ☐ Needs adjustment

- [ ] **Additional writing requirements:**
  - _____
  - **Decision:** ☐ Approved ☐ Needs adjustment

---

## 7. Stop Adding Experiments

### 7.1 Scope Freeze Confirmation
- [ ] **Confirm: stop adding new experiments, focus only on checklist items and writing**
  - **Decision:** ☐ Approved ☐ Exceptions noted below
  - **Notes:** _____

### 7.2 Exceptions (if any)
- [ ] **Exception 1:**
  - **Description:** _____
  - **Justification:** _____
  - **Decision:** ☐ Approved ☐ Rejected

- [ ] **Exception 2:**
  - **Description:** _____
  - **Justification:** _____
  - **Decision:** ☐ Approved ☐ Rejected

---

## 8. Additional Decisions & Notes

### 8.1 Open Questions
- **Question 1:** _____
  - **Answer:** _____
  
- **Question 2:** _____
  - **Answer:** _____

### 8.2 Supervisor Feedback
- **Feedback 1:** _____
- **Feedback 2:** _____
- **Feedback 3:** _____

### 8.3 Next Steps
- **Next meeting date:** _____
- **Communication channel:** _____
- **Immediate action items:**
  - [ ] _____
  - [ ] _____
  - [ ] _____

---

## Meeting Summary

**Key Decisions Made:**

1. _____
2. _____
3. _____

**Approved Scope:** _____

**Timeline:** _____

**Next Meeting:** _____

---

**Meeting Date:** _____  
**Attendees:** Diovandi Basheera Putra, Dr. Leonard P. Rusli  
**Completed by:** _____  
**Date completed:** _____

