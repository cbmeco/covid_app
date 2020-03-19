import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

delaware = 564696
chester = 519293
montgomery = 826075
bucks = 628341
philly = 1581000
S_default = delaware + chester + montgomery + bucks + philly
known_infections = 66

# Widgets
initial_infections = st.sidebar.number_input(
    "Currently Known Regional Infections", value=known_infections, step=10, format="%i"
)
#current_hosp = st.sidebar.number_input("Currently Hospitalized/Infected COVID-19 Comcast EEs or Deps", value=1, step=1, format="%i")
doubling_time = st.sidebar.number_input(
    "Doubling Time (days)", value=7, step=1, format="%i"
)
hosp_rate = (
    st.sidebar.number_input("Infected Rate when at 100%, Hospitalization % at 5%", 0, 100, value=100, step=1, format="%i")
    / 100.0
)
#icu_rate = (
    #st.sidebar.number_input("ICU %", 0, 100, value=2, step=1, format="%i") / 100.0
#)
#vent_rate = (
#    st.sidebar.number_input("Ventilated %", 0, 100, value=1, step=1, format="%i")
#    / 100.0
#)
hosp_los = st.sidebar.number_input("Length of Case", value=14, step=1, format="%i")
#icu_los = st.sidebar.number_input("ICU LOS", value=9, step=1, format="%i")
#vent_los = st.sidebar.number_input("Vent LOS", value=10, step=1, format="%i")
Penn_market_share = (
    st.sidebar.number_input(
        "Comcast EEs as Share of Regional Pop (%)", 0.0000, 100.0000, value=0.11, step=0.001, format="%f"
    )
    / 100.0
)
S = st.sidebar.number_input(
    "Regional Population", 0, 1000000000, value=12801989, step=100000, format="%i"
)
#current_hosp and slash before this if not working
#total_infections = Penn_market_share / hosp_rate
#detection_prob = initial_infections / total_infections

st.title("COVID-19 Impact Model for Epidemics: Adapted for use by Comcast Total Rewards")
st.markdown(
    """This tool can be used to model out the potential number of new cases of COVID-19 within a Comcast Employee & Member population based in a specific geographic areas.""")
st.markdown(
    """Use the inputs to the left to adjust the assumptions of the model to match the area of interest. The tool is defaulted to examine Pennsylvania. Below the first dropdown are assumptions to enter in for other locations""")
st.markdown(
    """*This tool was adapted for use at Comcast by the TRIP Data Science team. It was originally developed by Penn Medicine's Predictive Healthcare team. For questions and comments please see Chris Colameco (chris_colameco@comcast.com)"""
)

if st.checkbox("Show more info about this tool & Population Options"):
    st.subheader("""Initial Conditions for Largest Pockets of Comcast EE's""")
    st.markdown("""West Division: Known Infections as of 3/17: 1,946, Comcast EE's as share of pop: 0.02%, Regional Pop: 110,458,034""")
    st.markdown("""Northeast Division: Known Infections as of 3/17: 690, Comcast EE's as share of pop: 0.03%, Regional Pop: 63,924,546""")
    st.markdown("""Central Division: Known Infections as of 3/17: 792, Comcast EE's as share of pop: 0.03%, Regional Pop: 99,881,571""")

#    st.markdown("""Phila Metro Area: Known Infections (as of 3/16): 43, Comcast Pop: .44%, Regional Pop: 6,096,120""")
#    st.markdown("""NYC Metro Area: Known Infections (as of 3/16): 950, Comcast Pop: .07%, Regional Pop: 21,045,000""")
#    st.markdown("""Seattle Metro Area: Known Infections (as of 3/16): 769, Comcast Pop: .18%, Regional Pop: 3,939,363""")
#    st.markdown("""Atlanta Metro Area: Known Infections (as of 3/16): 37, Comcast Pop: .14%, Regional Pop: 5,949,951""")

    st.subheader(
        "[Discrete-time SIR modeling](https://mathworld.wolfram.com/SIRModel.html) of infections/recovery"
    )
    st.markdown(
        """The model consists of individuals who are either _Susceptible_ ($S$), _Infected_ ($I$), or _Recovered_ ($R$). 
The epidemic proceeds via a growth and decline process. This is the core model of infectious disease spread and has been in use in epidemiology for many years."""
    )
    
    st.markdown("""The dynamics are given by the following 3 equations.""")

    st.latex("S_{t+1} = (-\\beta S_t I_t) + S_t")
    st.latex("I_{t+1} = (\\beta S_t I_t - \\gamma I_t) + I_t")
    st.latex("R_{t+1} = (\\gamma I_t) + R_t")

    st.markdown(
        """To project the expected impact to the Comcast Population, we estimate the terms of the model. 
To do this, we use a combination of estimates from other locations, informed estimates based on logical reasoning, and best guesses from the American Hospital Association.
### Parameters
First, we need to express the two parameters $\\beta$ and $\\gamma$ in terms of quantities we can estimate.
- The $\\gamma$ parameter represents 1 over the mean recovery time in days. Since the CDC is recommending 14 days of self-quarantine, we'll use $\\gamma = 1/14$. 
- Next, the AHA says to expect a doubling time $T_d$ of 7-10 days. That means an early-phase rate of growth can be computed by using the doubling time formula:
""")
    st.latex("g = 2^{1/T_d} - 1")

    st.markdown("""
- Since the rate of new infections in the SIR model is $g = \\beta S - \\gamma$, and we've already computed $\\gamma$, $\\beta$ becomes a function of the initial population size of susceptible individuals.
$$\\beta = (g + \\gamma)/s$$
### Initial Conditions

- The initial number of infected will be the total number of confirmed cases in the area ({initial_infections}), divided by some detection probability to account for under testing {detection_prob:.2f}."""           
            
        
    )

# The SIR model, one time step
def sir(y, beta, gamma, N):
    S, I, R = y
    Sn = (-beta * S * I) + S
    In = (beta * S * I - gamma * I) + I
    Rn = gamma * I + R
    if Sn < 0:
        Sn = 0
    if In < 0:
        In = 0
    if Rn < 0:
        Rn = 0

    scale = N / (Sn + In + Rn)
    return Sn * scale, In * scale, Rn * scale


# Run the SIR model forward in time
def sim_sir(S, I, R, beta, gamma, n_days, beta_decay=None):
    N = S + I + R
    s, i, r = [S], [I], [R]
    for day in range(n_days):
        y = S, I, R
        S, I, R = sir(y, beta, gamma, N)
        if beta_decay:
            beta = beta * (1 - beta_decay)
        s.append(S)
        i.append(I)
        r.append(R)

    s, i, r = np.array(s), np.array(i), np.array(r)
    return s, i, r


## RUN THE MODEL

S, I, R = S, initial_infections/0.12, 0
#/ detection_prob


intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1

recovery_days = 14.0
# mean recovery rate, gamma, (in 1/days).
gamma = 1 / recovery_days

# Contact rate, beta
beta = (
    intrinsic_growth_rate + gamma
) / S  # {rate based on doubling time} / {initial S}


n_days = st.slider("Number of days to project", 30, 200, 60, 1, "%i")

beta_decay = 0.0
s, i, r = sim_sir(S, I, R, beta, gamma, n_days, beta_decay=beta_decay)


cases = i * hosp_rate * Penn_market_share
#icu = i * icu_rate * Penn_market_share
#vent = i * vent_rate * Penn_market_share

days = np.array(range(0, n_days + 1))
data_list = [days, cases#, icu, vent
             ]
data_dict = dict(zip(["day", "cases", "icu", "vent"], data_list))

projection = pd.DataFrame.from_dict(data_dict)

st.subheader("New Admissions - or # Infected if Hosp Rate changed to 100%")
st.markdown("Projected number of **daily** New cases of COVID-19 amongst Comcast EE's + Dependents in a given area")

# New cases
projection_admits = projection.iloc[:-1, :] - projection.shift(1)
projection_admits[projection_admits < 0] = 0

plot_projection_days = n_days - 10
projection_admits["day"] = range(projection_admits.shape[0])

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot(
    projection_admits.head(plot_projection_days)["cases"], ".-", label="Infected/Hospitalized"
)
#ax.plot(projection_admits.head(plot_projection_days)["icu"], ".-", label="ICU")
#ax.plot(projection_admits.head(plot_projection_days)["vent"], ".-", label="Ventilated")
ax.legend(loc=0)
ax.set_xlabel("Days from today")
ax.grid("on")
ax.set_ylabel("Daily New Cases")
st.pyplot()

admits_table = projection_admits[np.mod(projection_admits.index, 7) == 0].copy()
admits_table["day"] = admits_table.index
admits_table.index = range(admits_table.shape[0])
admits_table = admits_table.fillna(0).astype(int)

if st.checkbox("Show Projected Admissions in tabular form"):
    st.dataframe(admits_table)

st.subheader("Infected EE's + Deps (Census)")
st.markdown(
    "Projected **census** of COVID-19 patients, accounting for start and end of cases (length of infection = 14 days (estimated)"
)

# ALOS for each category of COVID-19 case (total guesses)

los_dict = {
    "cases": hosp_los#,
    #"icu": icu_los,
    #"vent": vent_los,
}

fig, ax = plt.subplots(1, 1, figsize=(10, 4))

census_dict = {}
for k, los in los_dict.items():
    census = (
        projection_admits.cumsum().iloc[:-los, :]
        - projection_admits.cumsum().shift(los).fillna(0)
    ).apply(np.ceil)
    census_dict[k] = census[k]
    ax.plot(census.head(plot_projection_days)[k], ".-", label="census")
    ax.legend(loc=0)

ax.set_xlabel("Days from today")
ax.grid("on")
ax.set_ylabel("Census")
st.pyplot()

census_df = pd.DataFrame(census_dict)
census_df["day"] = census_df.index
census_df = census_df[["day", "cases"#, "icu", "vent"
                       ]]

census_table = census_df[np.mod(census_df.index, 7) == 0].copy()
census_table.index = range(census_table.shape[0])
census_table.loc[0, :] = 0
census_table = census_table.dropna().astype(int)

if st.checkbox("Show Projected Census in tabular form"):
    st.dataframe(census_table)

#st.markdown(
#    """**Click the checkbox below to view additional data generated by this simulation**"""
#)
#if st.checkbox("Show Additional Projections"):
 #   st.subheader("The number of infected and recovered individuals in the hospital catchment region at any given moment")
  #  fig, ax = plt.subplots(1, 1, figsize=(10, 4))
   # ax.plot(i, label="Infected")
    #ax.plot(r, label="Recovered")
 #   ax.legend(loc=0)
 #   ax.set_xlabel("days from today")
 #   ax.set_ylabel("Case Volume")
 #   ax.grid("on")
 #   st.pyplot()

    # Show data
   # days = np.array(range(0, n_days + 1))
   # data_list = [days, s, i, r]
   # data_dict = dict(zip(["day", "susceptible", "infections", "recovered"], data_list))
   # projection_area = pd.DataFrame.from_dict(data_dict)
   # infect_table = (projection_area.iloc[::7, :]).apply(np.floor)
   # infect_table.index = range(infect_table.shape[0])

#if st.checkbox("Show Raw SIR Similation Data"):
#        st.dataframe(infect_table)

st.subheader("References & Acknowledgements")
st.markdown(
    """* AHA Webinar, Feb 26, James Lawler, MD, an associate professor University of Nebraska Medical Center, What Healthcare Leaders Need To Know: Preparing for the COVID-19
* We would like to recognize the valuable assistance in consultation and review of model assumptions by Michael Z. Levy, PhD, Associate Professor of Epidemiology, Department of Biostatistics, Epidemiology and Informatics at the Perelman School of Medicine 
    """
)
st.markdown("© 2020, The Trustees of the University of Pennsylvania")
