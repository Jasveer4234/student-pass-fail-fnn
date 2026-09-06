/**
 * Student Performance Intelligence - Interactive Frontend Logic
 * Lightweight, vanilla JavaScript for bidirectional slider synchronization,
 * live validation status checkmarks, What-If scenario simulation, and form resets.
 */

document.addEventListener("DOMContentLoaded", function () {
    // Configuration for input controls
    const controls = [
        {
            numId: "study_hours",
            rangeId: "study_hours_range",
            checkId: "check-sh",
            iconId: "icon-sh",
            textId: "text-sh",
            defaultVal: "7.5",
            min: 0,
            max: 24,
            unitLabel: "Study Hours (0–24 hrs)",
            decimals: 1,
        },
        {
            numId: "attendance",
            rangeId: "attendance_range",
            checkId: "check-att",
            iconId: "icon-att",
            textId: "text-att",
            defaultVal: "88.0",
            min: 0,
            max: 100,
            unitLabel: "Attendance (0–100%)",
            decimals: 1,
        },
        {
            numId: "previous_marks",
            rangeId: "previous_marks_range",
            checkId: "check-pm",
            iconId: "icon-pm",
            textId: "text-pm",
            defaultVal: "80.0",
            min: 0,
            max: 100,
            unitLabel: "Previous Marks (0–100)",
            decimals: 1,
        },
    ];

    function updateInputQualityStatus() {
        let allValid = true;

        controls.forEach((item) => {
            const numInput = document.getElementById(item.numId);
            const icon = document.getElementById(item.iconId);
            const text = document.getElementById(item.textId);
            if (!numInput) return;

            const val = parseFloat(numInput.value);
            const isValid = !isNaN(val) && val >= item.min && val <= item.max;

            if (isValid) {
                if (icon) {
                    icon.textContent = "✓";
                    icon.classList.remove("icon-invalid");
                }
                if (text) {
                    text.textContent = `${item.unitLabel} within valid range`;
                }
                numInput.classList.remove("input-invalid");
            } else {
                allValid = false;
                if (icon) {
                    icon.textContent = "✗";
                    icon.classList.add("icon-invalid");
                }
                if (text) {
                    text.textContent = `${item.unitLabel} is invalid (must be ${item.min}–${item.max})`;
                }
                numInput.classList.add("input-invalid");
            }
        });

        const stateBadge = document.getElementById("quality-state-badge");
        const footerNote = document.getElementById("quality-footer-note");

        if (stateBadge) {
            if (allValid) {
                stateBadge.textContent = "Valid";
                stateBadge.classList.remove("state-invalid");
            } else {
                stateBadge.textContent = "Action Needed";
                stateBadge.classList.add("state-invalid");
            }
        }

        if (footerNote) {
            footerNote.textContent = allValid
                ? "Ready for prediction."
                : "Please correct out-of-range inputs before proceeding.";
        }

        return allValid;
    }

    // 1. Setup bidirectional sync for each control
    controls.forEach((item) => {
        const numInput = document.getElementById(item.numId);
        const rangeInput = document.getElementById(item.rangeId);

        if (!numInput || !rangeInput) return;

        // If number input already has a value (e.g. preserved from server response), sync slider to it
        if (numInput.value && !isNaN(parseFloat(numInput.value))) {
            const val = parseFloat(numInput.value);
            if (val >= item.min && val <= item.max) {
                rangeInput.value = val;
            }
        } else if (!numInput.value && item.defaultVal) {
            numInput.value = item.defaultVal;
            rangeInput.value = item.defaultVal;
        }

        // When slider moves, update number field
        rangeInput.addEventListener("input", function () {
            numInput.value = parseFloat(this.value).toFixed(item.decimals);
            updateInputQualityStatus();
        });

        // When number input changes, update slider if within bounds
        numInput.addEventListener("input", function () {
            const val = parseFloat(this.value);
            if (!isNaN(val) && val >= item.min && val <= item.max) {
                rangeInput.value = val;
            }
            updateInputQualityStatus();
        });

        // Enforce bounds on blur
        numInput.addEventListener("blur", function () {
            const val = parseFloat(this.value);
            if (isNaN(val)) {
                this.value = item.defaultVal;
                rangeInput.value = item.defaultVal;
            } else if (val < item.min) {
                this.value = item.min.toFixed(item.decimals);
                rangeInput.value = item.min;
            } else if (val > item.max) {
                this.value = item.max.toFixed(item.decimals);
                rangeInput.value = item.max;
            }
            updateInputQualityStatus();
        });
    });

    // Initial validation check on load
    updateInputQualityStatus();

    // 2. Handle Form Submission & Loading State
    const form = document.getElementById("prediction-form");
    const predictBtn = document.getElementById("predict-btn");
    const btnText = document.getElementById("btn-text");

    if (form && predictBtn) {
        form.addEventListener("submit", function (e) {
            const isValid = updateInputQualityStatus();
            if (!isValid) {
                e.preventDefault();
                alert("Please ensure all inputs are within their valid bounds before submitting.");
                return;
            }

            // Set loading state
            predictBtn.disabled = true;
            if (btnText) {
                btnText.textContent = "Analyzing...";
            }
            predictBtn.classList.add("btn-loading");
        });
    }

    // 3. Handle Reset Button
    const resetBtn = document.getElementById("reset-btn");
    if (resetBtn) {
        resetBtn.addEventListener("click", function () {
            controls.forEach((item) => {
                const numInput = document.getElementById(item.numId);
                const rangeInput = document.getElementById(item.rangeId);
                if (numInput && rangeInput) {
                    numInput.value = item.defaultVal;
                    rangeInput.value = item.defaultVal;
                }
            });

            updateInputQualityStatus();

            const errorAlert = document.getElementById("error-alert");
            if (errorAlert) {
                errorAlert.style.display = "none";
            }

            const resultSection = document.getElementById("result-section");
            if (resultSection) {
                window.location.href = "/";
            }
        });
    }

    // 4. Interactive What-If Scenario Comparison Handler
    const compareWhatIfBtn = document.getElementById("compare-whatif-btn");
    const whatIfBox = document.getElementById("whatif-comparison-box");

    if (compareWhatIfBtn && whatIfBox) {
        compareWhatIfBtn.addEventListener("click", async function () {
            // Read alternative inputs
            const altHoursInput = document.getElementById("whatif_hours");
            const altAttInput = document.getElementById("whatif_att");
            const altMarksInput = document.getElementById("whatif_marks");

            const altHours = parseFloat(altHoursInput ? altHoursInput.value : 0);
            const altAtt = parseFloat(altAttInput ? altAttInput.value : 0);
            const altMarks = parseFloat(altMarksInput ? altMarksInput.value : 0);

            if (isNaN(altHours) || altHours < 0 || altHours > 24) {
                alert("Alternative Study Hours must be between 0 and 24 hours.");
                return;
            }
            if (isNaN(altAtt) || altAtt < 0 || altAtt > 100) {
                alert("Alternative Attendance must be between 0% and 100%.");
                return;
            }
            if (isNaN(altMarks) || altMarks < 0 || altMarks > 100) {
                alert("Alternative Previous Marks must be between 0 and 100.");
                return;
            }

            // Read current form inputs
            const currHours = parseFloat(document.getElementById("study_hours").value) || 7.5;
            const currAtt = parseFloat(document.getElementById("attendance").value) || 88.0;
            const currMarks = parseFloat(document.getElementById("previous_marks").value) || 80.0;

            compareWhatIfBtn.disabled = true;
            compareWhatIfBtn.textContent = "Evaluating Scenario...";

            try {
                // Perform asynchronous inference for both current and alternative via REST API
                const [resCurr, resAlt] = await Promise.all([
                    fetch("/api/predict", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ study_hours: currHours, attendance: currAtt, previous_marks: currMarks })
                    }),
                    fetch("/api/predict", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ study_hours: altHours, attendance: altAtt, previous_marks: altMarks })
                    })
                ]);

                const dataCurr = await resCurr.json();
                const dataAlt = await resAlt.json();

                if (!dataCurr.success || !dataAlt.success) {
                    alert("Error evaluating scenario: " + (dataCurr.error || dataAlt.error || "Unknown"));
                    return;
                }

                // Render Current Scenario details
                document.getElementById("scen-curr-hours").innerHTML = `Study Hours: <strong>${dataCurr.study_hours} hrs</strong>`;
                document.getElementById("scen-curr-att").innerHTML = `Attendance: <strong>${dataCurr.attendance}%</strong>`;
                document.getElementById("scen-curr-marks").innerHTML = `Previous Marks: <strong>${dataCurr.previous_marks}</strong>`;
                const currPill = document.getElementById("scen-curr-pill");
                currPill.textContent = `Prediction: ${dataCurr.prediction} (${dataCurr.confidence})`;
                currPill.style.color = dataCurr.prediction === "PASS" ? "var(--color-pass-accent)" : "var(--color-fail-accent)";

                // Render Alternative Scenario details
                document.getElementById("scen-alt-hours").innerHTML = `Study Hours: <strong>${dataAlt.study_hours} hrs</strong>`;
                document.getElementById("scen-alt-att").innerHTML = `Attendance: <strong>${dataAlt.attendance}%</strong>`;
                document.getElementById("scen-alt-marks").innerHTML = `Previous Marks: <strong>${dataAlt.previous_marks}</strong>`;
                const altPill = document.getElementById("scen-alt-pill");
                altPill.textContent = `Prediction: ${dataAlt.prediction} (${dataAlt.confidence})`;
                altPill.style.color = dataAlt.prediction === "PASS" ? "var(--color-pass-accent)" : "var(--color-fail-accent)";

                // Render Honest Comparison Interpretation
                const interpEl = document.getElementById("whatif-interpretation-text");
                if (interpEl) {
                    interpEl.textContent = `Under the alternative input scenario, the model predicts ${dataAlt.prediction} with ${dataAlt.confidence} confidence (Pass Probability: ${(dataAlt.pass_probability * 100).toFixed(2)}%).`;
                }

                whatIfBox.style.display = "flex";
            } catch (err) {
                console.error("Scenario evaluation failed:", err);
                alert("Failed to evaluate alternative scenario. Please verify connection.");
            } finally {
                compareWhatIfBtn.disabled = false;
                compareWhatIfBtn.innerHTML = `
                    <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        <polyline points="16 3 21 3 21 8"></polyline>
                        <line x1="4" y1="20" x2="21" y2="3"></line>
                        <polyline points="21 16 21 21 16 21"></polyline>
                        <line x1="15" y1="15" x2="21" y2="21"></line>
                        <line x1="4" y1="4" x2="9" y2="9"></line>
                    </svg>
                    <span>Compare Scenario</span>
                `;
            }
        });
    }
});
