# References and Source Notes

This file documents the competition context, team-provided source materials, upstream data references, and methodological documentation used by the GAIA project. It is separate from `CITATION.cff`, which tells visitors how to cite this repository.

## 1. Official competition references

1. **Wharton Global Youth Program, The Wharton School, University of Pennsylvania.** “Investment Competition.” Official competition overview, including the client-case format, WInS, required deliverables, and evaluation approach. Accessed October 30, 2026.  
   https://globalyouth.wharton.upenn.edu/competitions/investment-competition/

2. **Wharton Global Youth Program, The Wharton School, University of Pennsylvania.** “General Rules & Roles.” Official eligibility, advisor, deliverable, ethics, citation, and Generative AI rules. Accessed October 30, 2026.  
   https://globalyouth.wharton.upenn.edu/competitions/investment-competition/rules-roles/

3. **Wharton Global Youth Program.** “2026–2027 Wharton Global High School Investment Competition.” Registered-team SurveyMonkey Apply portal. The annual case study, detailed instructions, and deliverable materials are provided to registered teams through this portal.  
   https://wghsinvcomp.smapply.us/prog/2026-2027_wharton_global_high_school_investment_competition_/

## 2. Team-provided strategy and client materials

The following materials were supplied to the project by Commonwealth Funding. They are referenced here as project inputs; no claim is made that all of them are public documents.

4. **Wharton Global Youth Program.** *2026 WGY Gao Case Study—FINAL.* Team-provided client case PDF. Used for the contribution dates, ten beginning-of-year operating payments, 2031 communication point, reserve requirement, and prohibition on relying on outside operating funding.

5. **Commonwealth Funding.** *Wharton 2026 GAIA Strategy—Integrated 10-Module Master.* Team-provided strategy blueprint. Used for the end-to-end architecture, human-review governance, five-agent structure, uncertainty gate, GFSS/GCC, dynamic risk budgeting, and funding-probability workflow.

6. **Commonwealth Funding.** *Wharton 2026 GAIA Strategy Timeline and Team Division.* Team-provided project-governance document. Used for two-speed updating, human review, and the preference for interpretable models over unnecessary complexity.

7. **Commonwealth Funding.** *GAIA 3C/3D Workbook, October 14, 2026.* Team-provided research workbook. The workbook identifies its research as AI-simulated; consequently, imported records remain `human_approved=False` until independently reviewed.

8. **Commonwealth Funding.** *GAIA AI Signal Workbook, October 29, 2026.* Team-provided quarterly panel, account snapshot, positions, and strategy notes. Converted into editable project CSV inputs.

9. **Commonwealth Funding.** *allocation_12_plus_etf Workbook.* Team-provided strategic allocation inputs, including the 60% SPYM core and 40% individual-stock sleeve.

## 3. Upstream financial-data reference

10. **U.S. Securities and Exchange Commission.** “EDGAR Application Programming Interfaces.” Documentation for submissions and XBRL company-facts APIs. Accessed October 30, 2026.  
    https://www.sec.gov/search-filings/edgar-application-programming-interfaces

The team-supplied panel identifies SEC EDGAR XBRL and Sina Finance adjusted prices as upstream sources. This repository **does not independently re-download or re-verify every upstream record**. That provenance limitation must be retained when interpreting results.

## 4. Model and software documentation

11. **scikit-learn developers.** `sklearn.linear_model.Ridge`: L2-regularized linear least squares, including the `positive` coefficient constraint used by this project. Accessed October 30, 2026.  
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

12. **Pedregosa, F., et al.** “Scikit-learn: Machine Learning in Python.” *Journal of Machine Learning Research*, 12 (2011), 2825–2830.  
    https://jmlr.org/papers/v12/pedregosa11a.html

13. **NumPy developers.** NumPy documentation. Used for vectorized simulation and numerical analysis.  
    https://numpy.org/doc/

14. **pandas developers.** pandas documentation. Used for panel-data preparation and reporting.  
    https://pandas.pydata.org/docs/

15. **SciPy developers.** SciPy documentation. Used for rank-correlation diagnostics.  
    https://docs.scipy.org/doc/scipy/

## 5. Interpretation and citation cautions

- References 4–9 describe the provenance of team inputs; they do not establish independent verification of every claim in those inputs.
- Internal GAIA weights, thresholds, return assumptions, and scenario settings are research assumptions, not official Wharton parameters.
- Monte Carlo outputs are assumption-dependent scenario results, not guaranteed probabilities or promised facility costs.
- The offline narrative provider is deterministic extraction, not an LLM. Demo documents must not be cited as actual filings, earnings calls, or human-approved research.
- Anyone reusing company source URLs in `data/human_research.csv` should open the original document, check date and context, and cite the original publisher directly.
- Follow the official competition AI Policy: AI-generated material may not be submitted as your own work, and any AI-generated material included must be properly cited.
