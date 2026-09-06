"use strict";

document.addEventListener("DOMContentLoaded", () => {
    // --- Motion Preference ---
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // --- Current Student Inputs & Sliders ---
    const studyHoursInput = document.getElementById("study_hours");
    const studyHoursRange = document.getElementById("study_hours_range");
    const attendanceInput = document.getElementById("attendance");
    const attendanceRange = document.getElementById("attendance_range");
    const marksInput = document.getElementById("previous_marks");
    const marksRange = document.getElementById("previous_marks_range");

    // Header Badges for Current Student Inputs
    const studyBadgeVal = document.getElementById("study_hours_badge_val");
    const attBadgeVal = document.getElementById("attendance_badge_val");
    const marksBadgeVal = document.getElementById("previous_marks_badge_val");

    // --- Dynamic Profile Elements (Telemetry) ---
    const profileStudyBar = document.getElementById("profile-study-bar");
    const profileStudyVal = document.getElementById("profile-study-val");
    const profileStudyBadge = document.getElementById("profile-study-badge");

    const profileAttBar = document.getElementById("profile-att-bar");
    const profileAttVal = document.getElementById("profile-att-val");
    const profileAttBadge = document.getElementById("profile-att-badge");

    const profileMarksBar = document.getElementById("profile-marks-bar");
    const profileMarksVal = document.getElementById("profile-marks-val");
    const profileMarksBadge = document.getElementById("profile-marks-badge");

    // --- Input Quality Indicator ---
    const qualityDot = document.getElementById("quality-indicator-dot");
    const qualityText = document.getElementById("input-quality-text");
    const qualityBadge = document.getElementById("input-quality-badge");

    // --- Quick Presets ---
    const presetButtons = document.querySelectorAll(".preset-btn");

    // --- Buttons & Controls ---
    const predictForm = document.getElementById("prediction-form");
    const predictBtn = document.getElementById("predict-btn");

    // --- Error & Result Containers ---
    const formError = document.getElementById("form-error");
    const predictionEmptyState = document.getElementById("prediction-empty-state");
    const resultCard = document.getElementById("prediction-result-card");

    // --- Prediction Results Elements ---
    const resultDecisionBadge = document.getElementById("result-decision-badge");
    const resultPassProb = document.getElementById("result-pass-prob");
    const resultPassBar = document.getElementById("result-pass-bar");
    const resultFailProb = document.getElementById("result-fail-prob");
    const resultFailBar = document.getElementById("result-fail-bar");
    const resultConfidence = document.getElementById("result-confidence");
    const resultRiskLevel = document.getElementById("result-risk-level");
    const resultInterpretationText = document.getElementById("result-interpretation-text");
    const echoSh = document.getElementById("echo-sh");
    const echoAtt = document.getElementById("echo-att");
    const echoPm = document.getElementById("echo-pm");

    // --- Donut Chart Elements ---
    const donutPassRing = document.getElementById("donut-pass-ring");
    const donutFailRing = document.getElementById("donut-fail-ring");
    const donutCenterLabel = document.getElementById("donut-center-label");
    const donutCenterVal = document.getElementById("donut-center-val");
    const donutPassLegendVal = document.getElementById("donut-pass-legend-val");
    const donutFailLegendVal = document.getElementById("donut-fail-legend-val");

    // --- Performance Profile Elements ---
    const normStudyFill = document.getElementById("norm-study-fill");
    const normStudyText = document.getElementById("norm-study-text");
    const normAttFill = document.getElementById("norm-att-fill");
    const normAttText = document.getElementById("norm-att-text");
    const normMarksFill = document.getElementById("norm-marks-fill");
    const normMarksText = document.getElementById("norm-marks-text");

    const DONUT_CIRCUMFERENCE = 2 * Math.PI * 62; // ~389.557

    // --- Helper: Animate Counter ---
    function animateValue(element, start, end, duration, suffix = "", decimals = 2) {
        if (!element) return;
        if (prefersReducedMotion || duration <= 0) {
            element.textContent = end.toFixed(decimals) + suffix;
            return;
        }

        const startTime = performance.now();
        const diff = end - start;

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1.0);
            const ease = 1 - Math.pow(1 - progress, 3);
            const current = start + diff * ease;
            element.textContent = current.toFixed(decimals) + suffix;

            if (progress < 1.0) {
                requestAnimationFrame(update);
            } else {
                element.textContent = end.toFixed(decimals) + suffix;
            }
        }

        requestAnimationFrame(update);
    }

    // --- Validation Helpers ---
    function checkInputQuality(sh, att, marks) {
        const errors = [];
        let validCount = 0;

        if (!isNaN(sh) && sh >= 0 && sh <= 24) {
            validCount++;
        } else {
            errors.push("Study Hours must be 0–24 hrs");
        }

        if (!isNaN(att) && att >= 0 && att <= 100) {
            validCount++;
        } else {
            errors.push("Attendance must be 0–100%");
        }

        if (!isNaN(marks) && marks >= 0 && marks <= 100) {
            validCount++;
        } else {
            errors.push("Previous Marks must be 0–100");
        }

        return {
            isValid: validCount === 3,
            validCount: validCount,
            errors: errors
        };
    }

    function validateInputs(sh, att, marks) {
        if (isNaN(sh) || sh < 0 || sh > 24) {
            return "Study Hours must be a valid number between 0.0 and 24.0 hours.";
        }
        if (isNaN(att) || att < 0 || att > 100) {
            return "Attendance must be a valid percentage between 0.0% and 100.0%.";
        }
        if (isNaN(marks) || marks < 0 || marks > 100) {
            return "Previous Marks must be a valid score between 0.0 and 100.0 points.";
        }
        return null;
    }

    // --- Donut Ring Chart Renderer ---
    function renderDonutChart(passProb, failProb, decision, isPass) {
        if (!donutPassRing || !donutFailRing) return;

        donutPassRing.setAttribute("fill", "transparent");
        donutPassRing.setAttribute("stroke", "#10b981");
        donutPassRing.setAttribute("stroke-width", "16");

        donutFailRing.setAttribute("fill", "transparent");
        donutFailRing.setAttribute("stroke", "#ef4444");
        donutFailRing.setAttribute("stroke-width", "16");

        if (passProb === null || failProb === null) {
            donutPassRing.style.strokeDasharray = `0 ${DONUT_CIRCUMFERENCE}`;
            donutFailRing.style.strokeDasharray = `0 ${DONUT_CIRCUMFERENCE}`;
            if (donutCenterLabel) {
                donutCenterLabel.textContent = "READY";
                donutCenterLabel.style.fill = "#94a3b8";
            }
            if (donutCenterVal) {
                donutCenterVal.textContent = "--%";
                donutCenterVal.style.fill = "#94a3b8";
            }
            if (donutPassLegendVal) donutPassLegendVal.textContent = "--%";
            if (donutFailLegendVal) donutFailLegendVal.textContent = "--%";
            return;
        }

        const passLen = Math.max(0, Math.min(passProb * DONUT_CIRCUMFERENCE, DONUT_CIRCUMFERENCE));
        const failLen = Math.max(0, Math.min(failProb * DONUT_CIRCUMFERENCE, DONUT_CIRCUMFERENCE));

        donutPassRing.style.strokeDasharray = `${passLen} ${DONUT_CIRCUMFERENCE}`;
        donutPassRing.style.strokeDashoffset = "0";

        donutFailRing.style.strokeDasharray = `${failLen} ${DONUT_CIRCUMFERENCE}`;
        donutFailRing.style.strokeDashoffset = `${-passLen}`;

        if (donutCenterLabel) {
            donutCenterLabel.textContent = decision;
            donutCenterLabel.style.fill = isPass ? "var(--color-pass, #10b981)" : "var(--color-fail, #ef4444)";
        }
        if (donutCenterVal) {
            donutCenterVal.textContent = `${(passProb * 100).toFixed(2)}%`;
            donutCenterVal.style.fill = isPass ? "var(--color-pass, #10b981)" : "var(--color-fail, #ef4444)";
        }
        if (donutPassLegendVal) {
            donutPassLegendVal.textContent = `${(passProb * 100).toFixed(2)}%`;
        }
        if (donutFailLegendVal) {
            donutFailLegendVal.textContent = `${(failProb * 100).toFixed(2)}%`;
        }
    }

    // --- Performance Profile Renderer (Normalized Display Scale) ---
    function renderPerformanceProfile(sh, att, marks) {
        const shSafe = isNaN(sh) ? 0 : Math.max(0, Math.min(sh, 24));
        const attSafe = isNaN(att) ? 0 : Math.max(0, Math.min(att, 100));
        const marksSafe = isNaN(marks) ? 0 : Math.max(0, Math.min(marks, 100));

        const shNormPct = (shSafe / 24) * 100;
        const attNormPct = attSafe;
        const marksNormPct = marksSafe;

        if (normStudyFill) normStudyFill.style.width = `${shNormPct}%`;
        if (normStudyText) normStudyText.textContent = isNaN(sh) ? "-- hrs" : `${sh.toFixed(1)} hrs`;

        if (normAttFill) normAttFill.style.width = `${attNormPct}%`;
        if (normAttText) normAttText.textContent = isNaN(att) ? "--%" : `${att.toFixed(1)}%`;

        if (normMarksFill) normMarksFill.style.width = `${marksNormPct}%`;
        if (normMarksText) normMarksText.textContent = isNaN(marks) ? "-- / 100" : `${marks.toFixed(1)} / 100`;
    }

    // --- Dynamic Input Profile & Quality Visualizer ---
    function updateInputProfile() {
        const shRaw = studyHoursInput?.value;
        const attRaw = attendanceInput?.value;
        const marksRaw = marksInput?.value;

        const sh = parseFloat(shRaw);
        const att = parseFloat(attRaw);
        const marks = parseFloat(marksRaw);

        // Update Input Header Badges
        if (studyBadgeVal) {
            studyBadgeVal.textContent = isNaN(sh) ? "-- hrs" : `${sh.toFixed(1)} hrs`;
        }
        if (attBadgeVal) {
            attBadgeVal.textContent = isNaN(att) ? "--%" : `${att.toFixed(1)}%`;
        }
        if (marksBadgeVal) {
            marksBadgeVal.textContent = isNaN(marks) ? "-- / 100" : `${marks.toFixed(1)} / 100`;
        }

        // 1. Study Hours Telemetry (0-24 hrs)
        const shValSafe = isNaN(sh) ? 0 : sh;
        const shPct = Math.min(Math.max((shValSafe / 24) * 100, 0), 100);
        if (profileStudyBar) profileStudyBar.style.width = `${shPct}%`;
        if (profileStudyVal) profileStudyVal.textContent = isNaN(sh) ? "Invalid" : `${sh.toFixed(1)} hrs`;
        if (profileStudyBadge) {
            if (isNaN(sh) || sh < 0 || sh > 24) {
                profileStudyBadge.textContent = "Out of Range";
                profileStudyBadge.className = "profile-status-badge badge-low";
            } else if (sh < 3) {
                profileStudyBadge.textContent = "Low";
                profileStudyBadge.className = "profile-status-badge badge-low";
            } else if (sh < 7) {
                profileStudyBadge.textContent = "Moderate";
                profileStudyBadge.className = "profile-status-badge badge-moderate";
            } else {
                profileStudyBadge.textContent = "Strong";
                profileStudyBadge.className = "profile-status-badge badge-strong";
            }
        }

        // 2. Attendance Telemetry (0-100%)
        const attValSafe = isNaN(att) ? 0 : att;
        const attPct = Math.min(Math.max(attValSafe, 0), 100);
        if (profileAttBar) profileAttBar.style.width = `${attPct}%`;
        if (profileAttVal) profileAttVal.textContent = isNaN(att) ? "Invalid" : `${att.toFixed(1)}%`;
        if (profileAttBadge) {
            if (isNaN(att) || att < 0 || att > 100) {
                profileAttBadge.textContent = "Out of Range";
                profileAttBadge.className = "profile-status-badge badge-low";
            } else if (att < 60) {
                profileAttBadge.textContent = "Low";
                profileAttBadge.className = "profile-status-badge badge-low";
            } else if (att < 75) {
                profileAttBadge.textContent = "Moderate";
                profileAttBadge.className = "profile-status-badge badge-moderate";
            } else {
                profileAttBadge.textContent = "Good";
                profileAttBadge.className = "profile-status-badge badge-good";
            }
        }

        // 3. Previous Marks Telemetry (0-100 pts)
        const marksValSafe = isNaN(marks) ? 0 : marks;
        const marksPct = Math.min(Math.max(marksValSafe, 0), 100);
        if (profileMarksBar) profileMarksBar.style.width = `${marksPct}%`;
        if (profileMarksVal) profileMarksVal.textContent = isNaN(marks) ? "Invalid" : `${marks.toFixed(1)} / 100`;
        if (profileMarksBadge) {
            if (isNaN(marks) || marks < 0 || marks > 100) {
                profileMarksBadge.textContent = "Out of Range";
                profileMarksBadge.className = "profile-status-badge badge-low";
            } else if (marks < 40) {
                profileMarksBadge.textContent = "Needs Attention";
                profileMarksBadge.className = "profile-status-badge badge-low";
            } else if (marks < 70) {
                profileMarksBadge.textContent = "Developing";
                profileMarksBadge.className = "profile-status-badge badge-moderate";
            } else {
                profileMarksBadge.textContent = "Strong";
                profileMarksBadge.className = "profile-status-badge badge-strong";
            }
        }

        // 4. Input Quality Indicator
        const quality = checkInputQuality(sh, att, marks);
        if (quality.isValid) {
            if (qualityDot) qualityDot.className = "quality-dot dot-ready";
            if (qualityBadge) {
                qualityBadge.className = "quality-badge badge-ready";
                qualityBadge.textContent = "Ready";
            }
            if (qualityText) {
                qualityText.textContent = "Ready \u2022 3 of 3 indicators valid within expected ranges";
            }
        } else {
            if (qualityDot) qualityDot.className = "quality-dot dot-error";
            if (qualityBadge) {
                qualityBadge.className = "quality-badge badge-error";
                qualityBadge.textContent = "Check Inputs";
            }
            if (qualityText) {
                qualityText.textContent = `Check Inputs \u2022 ${quality.errors.join(", ")}`;
            }
        }

        // Update Performance Profile
        renderPerformanceProfile(sh, att, marks);

        // Check active preset
        syncActivePreset(sh, att, marks);
    }

    // --- Helper to sync preset active styling ---
    function syncActivePreset(sh, att, marks) {
        presetButtons.forEach((btn) => {
            const btnSh = parseFloat(btn.dataset.sh);
            const btnAtt = parseFloat(btn.dataset.att);
            const btnMarks = parseFloat(btn.dataset.marks);

            if (
                !isNaN(sh) && !isNaN(att) && !isNaN(marks) &&
                Math.abs(sh - btnSh) < 0.05 &&
                Math.abs(att - btnAtt) < 0.05 &&
                Math.abs(marks - btnMarks) < 0.05
            ) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });
    }

    // --- Bidirectional Input Sync Bindings ---
    function bindSync(numberEl, rangeEl, changeCallback) {
        if (!numberEl || !rangeEl) return;

        rangeEl.addEventListener("input", () => {
            numberEl.value = rangeEl.value;
            if (changeCallback) changeCallback();
        });

        numberEl.addEventListener("input", () => {
            const val = parseFloat(numberEl.value);
            if (!isNaN(val)) {
                rangeEl.value = val;
            }
            if (changeCallback) changeCallback();
        });
    }

    bindSync(studyHoursInput, studyHoursRange, updateInputProfile);
    bindSync(attendanceInput, attendanceRange, updateInputProfile);
    bindSync(marksInput, marksRange, updateInputProfile);

    // --- Presets Click Handler ---
    presetButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const targetSh = parseFloat(btn.dataset.sh);
            const targetAtt = parseFloat(btn.dataset.att);
            const targetMarks = parseFloat(btn.dataset.marks);

            if (studyHoursInput) studyHoursInput.value = targetSh.toFixed(1);
            if (studyHoursRange) studyHoursRange.value = targetSh;
            if (attendanceInput) attendanceInput.value = targetAtt.toFixed(1);
            if (attendanceRange) attendanceRange.value = targetAtt;
            if (marksInput) marksInput.value = targetMarks.toFixed(1);
            if (marksRange) marksRange.value = targetMarks;

            presetButtons.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            updateInputProfile();
        });
    });

    // --- Backend API Caller ---
    async function queryPredictionApi(studyHours, attendance, marks) {
        const payload = {
            study_hours: studyHours,
            attendance: attendance,
            previous_marks: marks
        };

        let response;
        try {
            response = await fetch("/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });
        } catch (networkErr) {
            throw new Error("Unable to reach the prediction service. Please verify that the application server is active.");
        }

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const msg = errorData.error || errorData.message || `Server returned error (${response.status})`;
            throw new Error(msg);
        }

        return await response.json();
    }

    // --- Render Main Prediction Result ---
    function displayCurrentResult(data) {
        if (!resultCard) return;

        // Hide Empty State, Reveal Result Card
        if (predictionEmptyState) {
            predictionEmptyState.classList.add("hidden");
        }
        resultCard.classList.remove("hidden");
        resultCard.setAttribute("aria-hidden", "false");

        const isPass = String(data.prediction).toUpperCase() === "PASS" || data.prediction === 1 || data.prediction_label === "Pass";
        const passProb = typeof data.pass_probability === "number" ? data.pass_probability : (parseFloat(data.pass_probability) || 0);
        const failProb = typeof data.fail_probability === "number" ? data.fail_probability : (parseFloat(data.fail_probability) || 0);

        const passProbPct = passProb * 100;
        const failProbPct = failProb * 100;

        // Parse confidence safely
        let confNumeric = 0;
        if (typeof data.confidence === "string") {
            confNumeric = parseFloat(data.confidence) || 0;
        } else if (typeof data.confidence === "number") {
            confNumeric = data.confidence > 1 ? data.confidence : data.confidence * 100;
        } else {
            confNumeric = isPass ? passProbPct : failProbPct;
        }

        // Decision Badge
        const decisionStr = isPass ? "PASS" : "FAIL";
        if (resultDecisionBadge) {
            resultDecisionBadge.className = `decision-badge ${isPass ? "badge-pass" : "badge-fail"}`;
            resultDecisionBadge.textContent = decisionStr;
        }

        // Render Donut Chart
        renderDonutChart(passProb, failProb, decisionStr, isPass);

        // Animate Pass / Fail Bars & Text
        if (resultPassBar) resultPassBar.style.width = `${passProbPct}%`;
        if (resultPassProb) animateValue(resultPassProb, 0, passProbPct, 600, "%", 2);

        if (resultFailBar) resultFailBar.style.width = `${failProbPct}%`;
        if (resultFailProb) animateValue(resultFailProb, 0, failProbPct, 600, "%", 2);

        if (resultConfidence) {
            animateValue(resultConfidence, 0, confNumeric, 600, "%", 2);
        }

        // Model-Estimated Risk Level
        if (resultRiskLevel) {
            if (passProb >= 0.75) {
                resultRiskLevel.textContent = "Lower Model-Estimated Risk";
                resultRiskLevel.className = "risk-value font-mono text-success";
            } else if (passProb >= 0.50) {
                resultRiskLevel.textContent = "Moderate Model-Estimated Risk";
                resultRiskLevel.className = "risk-value font-mono text-warning";
            } else {
                resultRiskLevel.textContent = "Higher Model-Estimated Risk";
                resultRiskLevel.className = "risk-value font-mono text-danger";
            }
        }

        // Model Interpretation Panel
        if (resultInterpretationText) {
            resultInterpretationText.textContent = `The model predicts ${decisionStr} for the supplied input values. Based on the current inputs, the estimated Pass probability is ${passProbPct.toFixed(2)}%. This is a model-based educational estimate, not a guaranteed outcome.`;
        }

        // Student Input Echo Summary
        const evaluatedSh = typeof data.study_hours === "number" ? data.study_hours : (parseFloat(studyHoursInput?.value) || 0);
        const evaluatedAtt = typeof data.attendance === "number" ? data.attendance : (parseFloat(attendanceInput?.value) || 0);
        const evaluatedPm = typeof data.previous_marks === "number" ? data.previous_marks : (parseFloat(marksInput?.value) || 0);

        if (echoSh) echoSh.textContent = `${evaluatedSh.toFixed(1)} hrs`;
        if (echoAtt) echoAtt.textContent = `${evaluatedAtt.toFixed(1)}%`;
        if (echoPm) echoPm.textContent = `${evaluatedPm.toFixed(1)} / 100`;

        // Update Performance Profile
        renderPerformanceProfile(evaluatedSh, evaluatedAtt, evaluatedPm);

        // Scroll result into view smoothly
        if (!prefersReducedMotion) {
            resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
    }

    // --- Handle Prediction Form Submit ---
    if (predictForm) {
        predictForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            if (formError) {
                formError.classList.add("hidden");
                formError.textContent = "";
            }

            const sh = parseFloat(studyHoursInput?.value);
            const att = parseFloat(attendanceInput?.value);
            const marks = parseFloat(marksInput?.value);

            const err = validateInputs(sh, att, marks);
            if (err) {
                if (formError) {
                    formError.textContent = err;
                    formError.classList.remove("hidden");
                }
                return;
            }

            // Set loading state
            if (predictBtn) {
                predictBtn.disabled = true;
                predictBtn.innerHTML = `<span class="spinner" aria-hidden="true"></span> <span>Analyzing Inputs...</span>`;
            }

            try {
                const data = await queryPredictionApi(sh, att, marks);
                displayCurrentResult(data);
            } catch (err) {
                if (formError) {
                    formError.textContent = err.message || "An unexpected error occurred during prediction.";
                    formError.classList.remove("hidden");
                }
            } finally {
                if (predictBtn) {
                    predictBtn.disabled = false;
                    predictBtn.innerHTML = `<span class="btn-text">Predict Academic Result</span><span class="btn-arrow" aria-hidden="true">&rarr;</span>`;
                }
            }
        });
    }

    // --- Mobile Sidebar Toggle & Navigation Anchor Smooth Scrolling ---
    const sidebar = document.getElementById("app-sidebar");
    const sidebarToggle = document.getElementById("sidebar-toggle");
    const sidebarOverlay = document.getElementById("sidebar-overlay");
    const navLinks = document.querySelectorAll(".nav-link");

    if (sidebarToggle && sidebar && sidebarOverlay) {
        sidebarToggle.addEventListener("click", () => {
            sidebar.classList.toggle("open");
            sidebarOverlay.classList.toggle("active");
        });

        sidebarOverlay.addEventListener("click", () => {
            sidebar.classList.remove("open");
            sidebarOverlay.classList.remove("active");
        });
    }

    // Active state highlighting and smooth navigation on link click
    navLinks.forEach((link) => {
        link.addEventListener("click", (e) => {
            const href = link.getAttribute("href");
            if (href && href.startsWith("#")) {
                const targetId = href.substring(1);
                const targetEl = document.getElementById(targetId);
                if (targetEl) {
                    e.preventDefault();
                    navLinks.forEach((l) => l.classList.remove("active"));
                    link.classList.add("active");

                    targetEl.scrollIntoView({ behavior: "smooth", block: "start" });

                    // Close mobile sidebar on link selection
                    if (sidebar && sidebarOverlay && window.innerWidth < 768) {
                        sidebar.classList.remove("open");
                        sidebarOverlay.classList.remove("active");
                    }
                }
            }
        });
    });

    // --- Initial Telemetry & Preset Sync on Page Load ---
    updateInputProfile();
});
