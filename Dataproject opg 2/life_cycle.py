"""
life_cycle.py

Simulating an income distribution from a simple life-cycle model, described in the assignment (2.1 The model).

Two shut-down style extensions are included, which makes it possible to re-simulate the model, 
while turning off individual mechanisms (Section 2.4):

the model also includes an optional disability-risk extension (Section 2.5). 
When pi_dis is greater than zero, a healthy individual of working age has a pi_dis probability
of becoming permanently disabled in each period. Once disabled, they receive a fixed disability
benefit y_dis for the rest of their working life.

"""

from __future__ import annotations                      #allows us to use types before they are defined and makes annotations behave more flexibly.
from dataclasses import dataclass, replace, field       #imports tools for working with data classes.
from typing import Optional                             #used for type annotations when something can either have a value or be None.
import numpy as np                                      #imports NumPy, a library for numerical calculations.
import pandas as pd                                     #imports Pandas, which is mainly used for working with tables of data.

EDUCATION_LABELS = ("short", "medium", "long")

#The following section defines a container for all the parameters of our life-cycle model. 
#The idea is that instead of having lots of separate variables, we keep them together in one Params object.

@dataclass
class Params:
    N: int = 50_000                     # number of individuals to simulate
    age_start: int = 18                 # Simulation starts at age 18
    age_end: int = 64                   # Simulation ends at age 64
    p_e: tuple = (0.40, 0.35, 0.25)     # probabilities of drawing each education level
    S_e: tuple = (1, 3, 5)              # number of years spent in education for each education level
    h0_e: tuple = (1.00, 1.20, 1.55)    # individual's starting human capital when they enter the labour market.
    D_e: tuple = (0.010, 0.020, 0.030)  # human-capital growth rates while employed, depending on education.
    delta: float = 0.06                 # human-capital depreciation rate while unemployed
    sigma_psi: float = 0.10             # controls the size of the random human-capital shock
    lam: float = 0.60                   # job-finding probability
    sigma: float = 0.05                 # job-separation probability
    y_SU: float = 0.45                  # the student grant
    rho: float = 0.60                   # fraction of last job's income received while unemployed
    y_floor: float = 0.35               # the minimum income received while unemployed and never having held a job
    seed: int = 2025                    # Random seed, controling the random-number generator used by the simulation

    # --- mechanism switches (Section 2.4) -------------------------------
    #When false, the model behaves as described in the assignment. 
    shut_education_hetero: bool = False     # When true, which education a person draws stops mattering
    shut_shocks: bool = False               # When true, human capital grows/shrinks smoothly with no randomness
    shut_depreciation: bool = False         # When true, human capital does not shrink while unemployed
    shut_unemployment: bool = False         # When true, everybody is employed once they enter the labor market

    # --- extension: disability risk (Section 2.5) ------------------------
    pi_dis: float = 0.0        # per-period probability of becoming disabled
    y_dis: float = 0.30        # disability benefit (flat, permanent)

    def with_(self, **kwargs) -> "Params":
        """Return a copy of the parameters with some fields overridden."""
        return replace(self, **kwargs)
    # makes a copy of 'params' with only the fields we specified changed, leaving the original untouched.


def simulate(params: Optional[Params] = None, **overrides) -> pd.DataFrame:
    if params is None:
        params = Params()                                       #if no settings were given, use the default settings
    if overrides:
        params = params.with_(**overrides)                      #if extra keyword tweaks were given, apply them on top

    rng = np.random.default_rng(params.seed)                    #create a reproducible random number generator
    N = params.N                                                #store the number of people to simulate
    ages = np.arange(params.age_start, params.age_end + 1)      #build the list of ages to simulate (18 to 64)
    T = len(ages)                                               #count how many years/periods that is

    p_e = np.array(params.p_e, dtype=float)     #convert the education probabilities
    S_e = np.array(params.S_e, dtype=float)     #convert the years-of-schooling values
    h0_e = np.array(params.h0_e, dtype=float)   #convert the starting human capital
    D_e = np.array(params.D_e, dtype=float)     #convert the human capital growth

    if params.shut_education_hetero:
        #do the following if this switch is turned on
        S_e = np.full(3, np.average(S_e, weights=p_e))      #replace the 3 schooling-length values with one shared average
        h0_e = np.full(3, np.average(h0_e, weights=p_e))    #replace the 3 starting-human-capital values with one shared average
        D_e = np.full(3, np.average(D_e, weights=p_e))      #replace the 3 growth-rate values with one shared average

    sigma_psi = 0.0 if params.shut_shocks else params.sigma_psi     #set shock size to 0 if shocks are switched off, otherwise use the normal value
    delta = 0.0 if params.shut_depreciation else params.delta       #set depreciation to 0 if depreciation is switched off, otherwise use the normal value

    # ---- draw education types ------------------------------------------------
    #gives every one of the N simulated people their own individual education, 
    #and looks up what that education means for them personally.
    e_idx = rng.choice(3, size=N, p=p_e)
    S_e_i = S_e[e_idx]
    h0_i = h0_e[e_idx]
    D_e_i = D_e[e_idx]

    # ---- state variables -------------------------------------------------
    #sets everyone's starting point before the simulation begins
    h = h0_i.copy()
    employed = np.zeros(N, dtype=bool)          # meaningful once in labour mkt
    ever_employed = np.zeros(N, dtype=bool)
    last_job_income = np.full(N, params.y_floor)
    disabled = np.zeros(N, dtype=bool)

    # ---- storage (later reshaped to a DataFrame) ------
    age_out = np.empty((T, N), dtype=int)       #to store each person's age every year
    in_educ_out = np.empty((T, N), dtype=bool)  #to store whether each person is in education every year
    employed_out = np.full((T, N), np.nan)      #to store employment status every year (blank while in school)
    disabled_out = np.full((T, N), np.nan)      #to store disability status every year (blank while in school)
    h_out = np.empty((T, N), dtype=float)       #to store human capital every year
    income_out = np.empty((T, N), dtype=float)  #to store income every year

    for t, age in enumerate(ages):
        #loop through each simulated year, one age at a time:
        in_labour_market = age >= (params.age_start + S_e_i)                    #check who has finished school and is working this year
        in_educ = ~in_labour_market                                             #ceveryone still in school this year
        newly_entering = in_labour_market & (age == params.age_start + S_e_i)   #people entering the labor market for the first time this year
        already_in_labour = in_labour_market & ~newly_entering                  #people who were already working before this year

        #Employment update:
        employed_new = employed.copy()
        employed_new[newly_entering] = False  #new graduates start out unemployed

        #Updates employment status for people already in the labor market:
        if already_in_labour.any():
            if params.shut_unemployment:
                employed_new[already_in_labour] = True
            else:
                draw = rng.random(N)
                was_emp = already_in_labour & employed
                was_unemp = already_in_labour & ~employed
                employed_new[was_emp] = draw[was_emp] < (1 - params.sigma)
                employed_new[was_unemp] = draw[was_unemp] < params.lam
        employed = employed_new

        # ---- disability extension: permanent absorbing state -------------
        #Every year, each healthy working person has a small random chance of becoming disabled. 
        #If it happens, they're marked disabled from then on
        if params.pi_dis > 0:
            can_become_disabled = in_labour_market & ~disabled
            newly_disabled = can_become_disabled & (
                rng.random(N) < params.pi_dis
            )
            disabled = disabled | newly_disabled
        #This splits everyone into the exact groups needed to decide their income: 
        #disabled, employed, never-worked-and-unemployed and worked-before-and-now-unemployed
        active = in_labour_market & ~disabled  
        dis_idx = in_labour_market & disabled
        emp_idx = active & employed
        unemp_idx = active & ~employed
        floor_idx = unemp_idx & ~ever_employed
        normal_unemp_idx = unemp_idx & ever_employed

        # ---- income --------------------------------------------------------
        #This works out everyone's income this year based on which group they're in
        #then updates each employed person's wage record for future reference.
        income = np.empty(N)
        income[in_educ] = params.y_SU
        income[dis_idx] = params.y_dis
        income[emp_idx] = h[emp_idx]
        income[floor_idx] = params.y_floor
        income[normal_unemp_idx] = params.rho * last_job_income[normal_unemp_idx]

        last_job_income[emp_idx] = h[emp_idx]
        ever_employed[emp_idx] = True

        # ---- store this period ---------------------------------------------
        #copies everything that happened this year into row t of the storage grids, building up the full history year by year.
        age_out[t] = age
        in_educ_out[t] = in_educ
        employed_out[t, in_labour_market] = employed[in_labour_market].astype(float)
        disabled_out[t, in_labour_market] = disabled[in_labour_market].astype(float)
        h_out[t] = h
        income_out[t] = income

        # ---- human-capital update for next period ---------------------------
        #This works out what happens to everyone's human capital going into next year
        psi = rng.lognormal(-0.5 * sigma_psi ** 2, sigma_psi, size=N)
        h_next = h.copy()
        h_next[emp_idx] = h[emp_idx] * (1 + D_e_i[emp_idx]) * psi[emp_idx]
        h_next[unemp_idx] = h[unemp_idx] * (1 - delta) * psi[unemp_idx]
        # education and disabled: human capital held fixed
        h = h_next

    # ---- assemble long-format panel --------------------------------------
    #This takes all the separate year×person grids we filled in during the simulation and flattens/stacks them
    #into one single table — one row per person per year — which is much easier to plot, filter, and analyze afterward.
    ids = np.tile(np.arange(N), T)
    edu_labels = np.array(EDUCATION_LABELS)[e_idx]

    df = pd.DataFrame(
        {
            "id": ids,
            "age": age_out.ravel(),
            "education": np.tile(edu_labels, T),
            "in_education": in_educ_out.ravel(),
            "employed": employed_out.ravel(),
            "disabled": disabled_out.ravel(),
            "human_capital": h_out.ravel(),
            "income": income_out.ravel(),
        }
    )
    return df

#measures how unequally income is spread across a group of people (gini coefficient):
def gini(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if np.any(x < 0):
        raise ValueError("gini() requires non-negative values")
    if x.sum() == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cum = np.cumsum(x_sorted)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n

#This function sorts people from poorest to richest, then tracks two things step by step: 
#what fraction of the population you've included so far, and what fraction of all income that group holds. 
#Plotting these two against each other gives the Lorenz curve
def lorenz_curve(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x_sorted = np.sort(x)
    n = x_sorted.size
    cum_income = np.cumsum(x_sorted)
    cum_income = np.concatenate(([0.0], cum_income / cum_income[-1]))
    pop_share = np.concatenate(([0.0], np.arange(1, n + 1) / n))
    return pop_share, cum_income

#a quick test, that only runs when you execute the file directly. 
#It runs the model once and prints three sanity checks
#  — education shares, unemployment rate, and overall inequality — 
#so you can quickly eyeball whether the model looks like it's behaving correctly.
if __name__ == "__main__":
    # quick smoke test when run directly: python life_cycle.py
    df = simulate()
    print(df.groupby("education")["id"].nunique() / df["id"].nunique())
    working = df[~df["in_education"]]
    print("Unemployment rate:", 1 - working["employed"].mean())
    print("Gini (pooled):", gini(df["income"].values))
