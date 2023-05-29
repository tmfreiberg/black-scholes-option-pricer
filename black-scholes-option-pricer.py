## BLACK-SCHOLES OPTIONS PRICER INCLUDING IMPLIED VOLATILITY PLOTS, HEATMAPS, ETC.

## Save this as foo.py and run it with 'streamlit run foo.py [-- script args]'.
## Alternatively, run 'python -m streamlit run foo.py'.

## LIBRARIES

import streamlit as st # For the user interface. Everything the user will interact with is based on a Streamlit method. 
import numpy as np # We'll obviously do some calculations.
from scipy.stats import norm # To compute option price, we use the CDF of a normal distribution. Used in 'call' and 'put' functions below.
from scipy.optimize import fsolve # To compute implied volatility. 
import matplotlib.pyplot as plt # To do plots for implied volatility, spot vs premium time contours, and heatmaps.
import pandas as pd # To display data retrieved from a table in a nice dataframe.
import sqlite3 # Data will be stored in and retrieved from tables in a database.

## FUNCTIONS

## BLACK-SCHOLES EUROPEAN OPTION PRICE

## PARAMETERS
## S - spot price
## K - strike price
## r - annualized risk-free interest rate (which we ask the user to enter as a percentage)
## tau - time to maturity in years
## sigma - volatility of underlying asset

## DEFINE CALL AND PUT FUNCTIONS

def call(S, K, r, tau, sigma):
    if tau == 0:
        return max(S - K, 0)
    else:
        d_1 = (1/(sigma*(np.sqrt(tau))))*(np.log(S/K) + (r + sigma**2/2)*(tau))
        d_2 = d_1 - sigma*np.sqrt(tau)
        # norm.cdf is standard normal CDF
        return norm.cdf(d_1)*S - norm.cdf(d_2)*K*np.exp(-r*(tau)) 

def put(S, K, r, tau, sigma):
    if tau == 0:
        return max(K - S, 0)
    else:
        d_1 = (1/(sigma*(np.sqrt(tau))))*(np.log(S/K) + (r + sigma**2/2)*(tau))
        d_2 = d_1 - sigma*np.sqrt(tau)
        # norm.cdf is standard normal CDF
        return norm.cdf(-d_2)*K*np.exp(-r*(tau)) - norm.cdf(-d_1)*S

## A FUNCTION THAT COMPUTES IMPLIED VOLATILITY (IV)

## An auxiliary function used in the IV computation
    
def extreme_call_option_price(S,K,r,tau):
    # A volatility of less than volmin is quite small, where
    volmin = 0.0001
    # The corresponding premium is 
    pmin = call(S,K,r,tau,volmin)
    # At some point, a non-negligible increase in volatility results in a negligible increase in the premium.
    # Let's say that, if an increase in volatility of vstep results in an increase in the premium of less than pstep,
    # then the premium is too high (or possibly too low???).
    volmax = volmin
    vstep = 1
    pstep = 0.01
    loop_limit = 1000
    loop = 0
    while (call(S,K,r,tau,volmax + vstep) - call(S,K,r,tau,volmax) >= pstep and loop < loop_limit + 1):
        loop += 1
        volmax += vstep
    volmax = volmax - vstep
    # We have not ruled out the possibility that volmax = volmin yet.
    # If that is the case, it means that 
    # call(S,K,r,tau,volmin + loop_limit*vstep) - call(S,K,r,tau,volmin) < pstep.
    # This seems quite unlikely. Still...
    if volmax == volmin:
        return 'Try some other parameters.'
    volmax = volmax - volmin # Just so volmax is an integer, rather than something like 10.0001...
    pmax = call(S,K,r,tau,volmax)
    return volmin, pmin, volmax, pmax, vstep, pstep

## And here is the IV function.

def implied_volatility(S,K,r,tau,p): # p is option price ['p' for 'premium'].
    extreme = extreme_call_option_price(S, K, r, tau)
    if type(extreme) != str:
        volmin, pmin, volmax, pmax, vstep, pstep = extreme_call_option_price(S, K, r, tau)
    if p < pmin:
        return f"Volatility decreases with option price, and volatility is already as low as {100*volmin}% when option price is {pmin:.2f}."
    if p > pmax:
        return f"When option price equals {pmax:.2f}, volatility equals {100*volmax}%, and an increase of less than {pstep} in the option price leads to an increase of more than {100*vstep}% in volatility."
    if pmin <= p <= pmax:
        def isolate_sigma(sigma):
            return call(S,K,r,tau,sigma) - p
        IV = fsolve(isolate_sigma,(volmin + volmax)/2)[0]
#         # Or we could do things from first principles, without using fsolve...
#         IV = volmin
#         precision = 5 # Can modify this to n where we want n digits 
#         loop_limit = 1000
#         for d in range(precision + 1): # 
#             step = 10**(-d) 
#             loop = 0 
#             while (call(S,K,r,tau,IV) < p and loop < loop_limit + 1):
#                 loop += 1
#                 IV += step
#             IV = IV - step # This will always be a little bit less than the true volatility.
        return IV

## DEFINE UPDATE FUNCTION FOR WHEN WIDGETS ARE CHANGED

## In particular, any time either a number_input widget or a slider widget is changed, we want agreement between number_input widget, slider widget, and values used to calculate option price.
## We have four tabs. We could handle all widgets over all tabs with a single function. However, we will do one update function per tab, as not all tabs have both sliders and number input widgets.
## Also, it is not necessary to update all paramaters across all tabs when we only work in one tab at a time.
## It probably doesn't make much difference either way, in terms of the amount of time taken to update the parameters.

## Update function for first tab. See below for parameter_dict.
## It basically contains three versions of S, K, r, tau, sigma: one used to compute call/put option price, one for the number input widget, and one for the slider widget.
## Suffix '_val' for the variable plugged into our call/put functions; suffix '_box' for the number input widgets (because they appear as a box to the user); suffix '_slider' for the slider input widget.
## We want all three to agree every time a widget is updated: that's what the update function does.

def update():
    for k in parameter_dict.keys():
        k_box, k_slider, k_val = k+'_box', k+'_slider', k+'_val'
        # This condition (if true) tells us that the number_input widget for k has been changed.
        if st.session_state[k_box] != st.session_state[k_val]:
            # Thus, we update k_val to reflect this.
            st.session_state[k_val] = st.session_state[k_box]
            st.session_state[k_slider] = st.session_state[k_val]
        # On the other hand, if a slider widget for k has been changed...
        elif st.session_state[k_slider] != st.session_state[k_val]:
            st.session_state[k_val] = st.session_state[k_slider]
            st.session_state[k_box] = st.session_state[k_val]

## Similar to the above, but for the second tab. Everything is more-or-less the same as in the first tab in terms of widgets and parameters.
## Everything related to the implied volatility tab is prefixed with 'IV'.

def IVupdate():
    for k in IVparameter_dict.keys():
        k_box, k_slider, k_val = k+'_box', k+'_slider', k+'_val'
        if st.session_state[k_box] != st.session_state[k_val]:
            st.session_state[k_val] = st.session_state[k_box]
            st.session_state[k_slider] = st.session_state[k_val]
        elif st.session_state[k_slider] != st.session_state[k_val]:
            st.session_state[k_val] = st.session_state[k_slider]
            st.session_state[k_box] = st.session_state[k_val]

## Similar to the above, but for the third tab. Everything is more-or-less the same as in the first and second tabs in terms of widgets and parameters.
## Everything related to the 'spot vs premium' tab is prefixed with 'SP'. See below for what we plot in this tab.

def SPupdate():
    for k in SPparameter_dict.keys():
        k_box, k_slider, k_val = k+'_box', k+'_slider', k+'_val'
        if st.session_state[k_box] != st.session_state[k_val]:
            st.session_state[k_val] = st.session_state[k_box]
            st.session_state[k_slider] = st.session_state[k_val]
        elif st.session_state[k_slider] != st.session_state[k_val]:
            st.session_state[k_val] = st.session_state[k_slider]
            st.session_state[k_box] = st.session_state[k_val]

## Similar to the above, but for the fourth tab. For the heatmap tab, everything is prefixed with 'HM'.
## In this tab, we just have number input widgets, and no slider widgets.            

def HMupdate():
    for k in HMparameter_dict.keys():
        k_box, k_val = k+'_box', k+'_val'
        if st.session_state[k_box] != st.session_state[k_val]:
            st.session_state[k_val] = st.session_state[k_box]           

## DEFINE UPDATE FUNCTION FOR SPOT VS PREMIUM CONTOUR PLOT IN SPOT VS PREMIUM TAB
            
## In this tab we plot spot vs premium at time tau = 0 years by default.
## The user can then add plots for different values of tau, by selecting tau and clicking 'Ass contour'.
## We just keep a list of the tau, and append the new tau to the list: that's what this function does.
## Then we use the updated list in the matplotlib plot.

def SPupdate_contours():
    SPtau = SPtau_years + SPtau_months/12 + SPtau_weeks/52 + SPtau_days/365
    if SPtau not in st.session_state['SPtau_contours']:        
        st.session_state['SPtau_contours'].append(SPtau)

## DEFINE UPDATE FUNCTIONS FOR HEATMAP TAB

## We have a button that allows the user to toggle between displaying a database table and not displaying it.
## The database table contains a history of parameter choices for which the user has saved data to the database.
## We use the session state dictionary for this: the defaul value for the 'HMdb_history' key is 'hide'. 

def HMshow_db_history():
    if st.session_state['HMdb_history'] == 'hide':
        st.session_state['HMdb_history'] = 'show'
    else: 
        st.session_state['HMdb_history'] = 'hide'

## We give the user the option to clear the database completely. There will be two related tables in the database.
## Clicking 'Clear database' will drop both.

def HMclear_database():
    conn = sqlite3.connect('options_price_db')
    conn.execute('DROP TABLE IF EXISTS HMhistory_table')
    conn.execute('DROP TABLE IF EXISTS HMpremium_table')
    conn.execute('DROP TABLE IF EXISTS HMdelta_premium_table')
    conn.commit()
    conn.close()

## What follows is the most involved update function. When the user clicks 'Update database', we update two related tables in our databse.
## The first is the table containing the parameter history, mentioned above.
## The second is the matrix on which the heatmap is based. This can also be displayed in a dataframe.

def HMupdate_database():
##    We'll have two related tables.
##    Table 'HMhistory_table' will contain a history of the parameter choices (HMS,HMK,HMr,HMtau,HMsigma) for which the user has generated a heatmap/dataframe and saved data to the database.
##    Table 'HMpremium_table' will contain all of the data needed to construct a heatmap/dataframe, corresponding to a given choice of parameters. See below for details.
 
##    STEP 1. Create table in database if it does not already exist, then search database for table with data corresponding to current parameters.
##    STEP 2. If data corresponding to current parameters is found, do nothing.
##    STEP 3. If not, generate an array containing the data (the heatmap and/or dataframe will be based on this new array), and
##    STEP 4. insert this newly generated array into our database table. Note that the new data array will not have the same dimensions as the table.

##    STEP 1. Create a table/search database for table with data.
##    Open a connection to our database and create tables (in case they do not already exist). Don't forget to commit!       
    conn = sqlite3.connect('options_price_db')
    conn.execute('CREATE TABLE IF NOT EXISTS HMhistory_table (spot number, strike number, interest number, time number, years int, months int, weeks int, days int, volatility number, premium, PRIMARY KEY(spot, strike, interest, time, months, weeks, days, volatility))')
    conn.executemany('INSERT OR IGNORE INTO HMhistory_table VALUES(?,?,?,?,?,?,?,?,?,?)', [(HMS, HMK, HMrpc, HMtau, HMtau_years, HMtau_months, HMtau_weeks, HMtau_days, HMsigmapc, HMp)])#, delta_volatility, delta_spot, delta_premium
##    That's it for the HMhistory_table. Everything that follows is for the HMpremium_table.    
    conn.execute('CREATE TABLE IF NOT EXISTS HMpremium_table (spot number, strike number, interest number, time number, years int, months int, weeks int, days int, volatility number, delta_volatility number, delta_spot number, delta_premium number, FOREIGN KEY(spot, strike, interest, years, months, weeks, days, volatility) REFERENCES HMhistory_table(spot, strike, interest, years, months, weeks, days, volatility))')
    conn.commit()
##    Our HMpremium_table has 12 columns.
##    We've ensured no row will ever be duplicated with the FOREIGN KEY/PRIMARY KEY constraint (referencing the HMhistory_table).    
##    The first nine contain the current parameters (HMS, HMK, HMr, HMtau, HMtau_years, HMtau_months, HMtau_weeks, HMtau_days, HMsigma) we want to use to generate the heatmap etc. (Of course, HMtau is calculated form the user's input for years, months, weeks, days, and HMtau is all we need for the call option premium calculation.)
##    The last three respectively contain the changes in volatility, the changes in spot prices, and the resulting change in premium.
##    We only need to extract the last three columns. Ensure they are ordered, first by change in volatility (descending), then by change in spot price (ascending).

##    Query for table entries, create cursor, and fetch the result of the query, store it in a variable called found_rows (a list).
    res = conn.execute("SELECT delta_volatility, delta_spot, delta_premium FROM HMpremium_table WHERE (spot, strike, interest, years, months, weeks, days, volatility) =(?,?,?,?,?,?,?,?) ORDER BY delta_volatility DESC, delta_spot ASC", (HMS,HMK,HMr,HMtau_years,HMtau_months,HMtau_weeks,HMtau_days,HMsigma))
    c = conn.cursor()
    found_rows = res.fetchall()
    c.close()
##    We can close the cursor as we won't need it again, but we'll leave our connection open for now.        
##    If found_rows has length zero, we have no data: go to STEP 3. Otherwise, continue to STEP 2.
    
##    STEP 2. If found, do nothing. If not found, go to STEP 3.
    if len(found_rows) > 0:
        conn.close()
        pass
    
##    STEP 3. We found no data in our table corresponding to (HMS,HMK,HMr,HMtau,HMsigma), so we create it.
##    We construct an array with HMnum_rows rows and HMnum_cols columns. We decide what they are here: the user cannot change these values.
##    The rows will correspond to volatilities changes [v0,v1,...] evenly spaced between two values that depend on what the user chose for HMsigma.
##    The columns will correspond to spot price changes [s0,s1,...] evenly spaced between two values related to the user's input for HMS and their choise of HMsigma.
##    Entry (i,j) of the array is equal to the change call option price corresponding to (vi,sj), and HMK, HMr, HMtau (fixed).
##    i.e. call(HMS + si, HMK, HMr, HMtau, HMsigma + vj) - call(HMS, HMK, HMr, HMtau, HMsigma)
    else:
        HMnum_rows = 20 # Can increase to increase the fine-ness of the heatmap, or decrease to make more coarse.
        HMnum_cols = 20  
        # Similarly for the range for volatility changes. Note that the order is decreasing.
        HMrows = np.linspace(2*HMsigmapc - HMsigmapc,(HMsigmapc/2) - HMsigmapc,HMnum_cols)
        # The below range for the spot price changes seems reasonable: if the user wants to see other results, they can adjust HMS and/or HMsigma.
        HMcols = np.linspace(max(0,HMS*(1 - 2*HMsigma)) - HMS,HMS*(1 + 2*HMsigma) - HMS,HMnum_rows)
        # Create an array of the right dimensions.
        HMpremium_matrix = np.empty((HMnum_rows,HMnum_cols), dtype=np.float64, order='C')
        # Now fill in the array with the premiums.
        for i in range(HMnum_rows):
            for j in range(HMnum_cols):
                vpc, delta_S = HMrows[i], HMcols[j]
                delta_sigma = vpc/100 # Recall that volatility is input/displayed as percentage
                delta_p = call(HMS + delta_S,HMK,HMr,HMtau,HMsigma + delta_sigma) - call(HMS,HMK,HMr,HMtau,HMsigma)
                HMpremium_matrix[i][j] = delta_p
                              
##    STEP 4. Now we insert the data from the array into a table in our database.
##    We have (say) an (m x n) array, from which we want to create a sub-table of (mn x 3) rows.
##    First column corresponds to change in volatility (m values), second column to change in spot price (n values), third column change in premium (mn values).
##    For example, if m = 2 and n = 3 with volatilities v0, v1 and spots s0, s1, s2, then we'd have an array of changes in premium: [[p00,p01,p02],[p10,p11,p12]].
##    We want to create a table like this: [[v0,s0,p00], [v0,s1,p01], [v0,s2,p02], [v1,s0,p10], [v1,s1,p11], [v1,s2,p12]].
##    There will be mn rows inserted. The first n rows have v0 (say) in the delta_volatility column, and s0,s1,...,s_{n-1} (say) in the delta_spot column.
##    The corresponding change in premium in row j is entry (0,j) from our (m x n) array heatmap.
##    The next n rows have v1 (say) in the delta_volatility column, and s0,s1,...,s_{n-1} again in the delta_spot column. And so on...
##    So in the ith row, say i = qn + r, 0 <= r < n. Then we want delta_volatility to be v_q and delta_spot to be s_r (and delta_premium to be entry (q,r) of our array).
        for i in range(HMnum_rows*HMnum_cols):
            row_to_insert = [(HMS,HMK,HMr,HMtau,HMtau_years,HMtau_months,HMtau_weeks,HMtau_days,HMsigma,HMrows[i//HMnum_cols],HMcols[i%HMnum_rows],HMpremium_matrix[i//HMnum_cols][i%HMnum_rows])]
            conn.executemany('INSERT OR IGNORE INTO HMpremium_table VALUES(?,?,?,?,?,?,?,?,?,?,?,?)', row_to_insert)
##    Commit the insertion and close the connection.
        conn.commit()
        conn.close()

## DEFAULT VALUES FOR ALL PARAMETERS.

## We'll create four dictionaries, one for each tab. Essentially, they all contain the S, K, r, tau, sigma parameters.
## Variables for implied volatility tab are prefixed by IV; spot vs premium tab by SP; heatmap tab by HM.
## We let the user input time by selecting years, months, weeks, days, then we convert it to number-of-years single value.
## We let the user input the interest and volatility as percentages, then we divide by 100 before inputting into call/put functions.
## Hence the suffixes _years, _months, etc., and pc (percentage).
## Our default values for S, K, r, tau, sigma are 100, 100, 0.05, 1, 0.2 respectively.
       
parameter_dict = { 
        'S' : 100.00,
        'K' : 100.00,
      'rpc' : 5.00, 
      'tau_years' : 1, 
     'tau_months' : 0, 
      'tau_weeks' : 0, 
       'tau_days' : 0, 
    'sigmapc' : 20.00}

## In he implied volatility tab, the user selects the spot, strike, interest, time to maturity, and call option premium, and we compute the implied volatility.
## p for premium...

IVparameter_dict = {
        'IVS' : 100.00, 
        'IVK' : 100.00, 
      'IVrpc' : 5.00, 
      'IVtau_years' : 1,
     'IVtau_months' : 0, 
      'IVtau_weeks' : 0, 
       'IVtau_days' : 0, 
        'IVp' : call(100,100,5.00/100,1,20.00/100)}

## In the spot vs premium tab, the user selects strike, interest, time to maturity, and we plot spot vs premium for a range of spot prices.
## The user then adds to the plot with different time values: it's a contour plot of the call option surface.

SPparameter_dict = {
        'SPK' : 100.00,
      'SPrpc' : 5.00, 
      'SPtau_years' : 1,
     'SPtau_months' : 0,
      'SPtau_weeks' : 0, 
       'SPtau_days' : 0, 
    'SPsigmapc' : 20.00}

## In the heatmap tab, the user enters spot, strike, interest, time to maturity, and volatility, then we generate a heatmap etc.

HMparameter_dict = {
        'HMS' : 100.00,
        'HMK' : 100.00,
      'HMrpc' : 5.00,
      'HMtau_years' : 1,
     'HMtau_months' : 0,
      'HMtau_weeks' : 0,
       'HMtau_days' : 0,
    'HMsigmapc' : 20.00,
        'HMp' : call(100,100,5.00/100,1,20.00/100)}

## The above variables are what we use to do all the calculations.
## We also have widgets corresponding to these variables (that's how the user selects the values of the variables).
## So we have two to three other variables for each variable, suffixed by '_val', '_box', and '_slider'.
## These are really sesstion state dictionary keys, and go into the update functions. Then the '_val' key values plugged into the variables above.

## The first three tabs have both number input and slider widgets. The fourth tab (heatmap tab) just has number input widgets.

parameter_dict_box_and_slider_tabs = {**parameter_dict, **IVparameter_dict, **SPparameter_dict}

for k in parameter_dict_box_and_slider_tabs.keys(): 
    k_val, k_box, k_slider = k+'_val', k+'_box', k+'_slider'
    if k_val not in st.session_state.keys():
        st.session_state[k_val] = parameter_dict_box_and_slider_tabs[k]
    if k_box not in st.session_state.keys():
        st.session_state[k_box] = parameter_dict_box_and_slider_tabs[k]
    if k_slider not in st.session_state.keys():
        st.session_state[k_slider] = parameter_dict_box_and_slider_tabs[k]

for k in HMparameter_dict.keys(): 
    k_val, k_box = k+'_val', k+'_box'
    if k_val not in st.session_state.keys():
        st.session_state[k_val] = HMparameter_dict[k]
    if k_box not in st.session_state.keys():
        st.session_state[k_box] = HMparameter_dict[k]

## We also use the session state dictionary to keep track of the contours in the spot vs premium tab, and the 'Show/hide history' button in the heatmap tab...
        
if 'SPtau_contours' not in st.session_state.keys():
    st.session_state['SPtau_contours'] = [0]  

if 'HMdb_history' not in st.session_state.keys():
    st.session_state['HMdb_history'] = 'hide'    
       
## Finally, all the variables that are actually plugged in to our call/put functions, etc.
## When a widget is updated, the relevant update function is called, updating widget values and '_val' values.
## Then, the corresponding variables below are updated, and plugged into our call/put functions, etc.

## Variables used to calculate option price.
S = st.session_state['S_val']
K = st.session_state['K_val']
rpc = st.session_state['rpc_val'] # We won't use this one other than for the value in the number_input and slider widgets for interest.
r = rpc/100
tau_years = st.session_state['tau_years_val'] 
tau_months = st.session_state['tau_months_val']
tau_weeks = st.session_state['tau_weeks_val']
tau_days = st.session_state['tau_days_val']
tau = tau_years + tau_months/12 + tau_weeks/52 + tau_days/365 # This is the value we plug into the call and put option price formulas
sigmapc = st.session_state['sigmapc_val'] # We won't use this one other than for the value in the number_input and slider widgets for volatility.
sigma = sigmapc/100

## Variables used to calculate implied volatility in the second tab.
IVS = st.session_state['IVS_val']
IVK = st.session_state['IVK_val']
IVrpc = st.session_state['IVrpc_val'] # We won't use this one other than for the value in the number_input and slider widgets for interest.
IVr = IVrpc/100
IVtau_years = st.session_state['IVtau_years_val'] 
IVtau_months = st.session_state['IVtau_months_val']
IVtau_weeks = st.session_state['IVtau_weeks_val']
IVtau_days = st.session_state['IVtau_days_val']
IVtau = IVtau_years + IVtau_months/12 + IVtau_weeks/52 + IVtau_days/365 # This is the value we plug into the call option price formula
IVp = st.session_state['IVp_val']

## Variables used to calculate call option price for the contour plot of the third tab.
SPK = st.session_state['SPK_val']
SPrpc = st.session_state['SPrpc_val'] # We won't use this one other than for the value in the number_input and slider widgets for interest.
SPr = SPrpc/100
SPtau_years = st.session_state['SPtau_years_val'] 
SPtau_months = st.session_state['SPtau_months_val']
SPtau_weeks = st.session_state['SPtau_weeks_val']
SPtau_days = st.session_state['SPtau_days_val']
SPtau = SPtau_years + SPtau_months/12 + SPtau_weeks/52 + SPtau_days/365 # This is the value we plug into the call and put option price formulas
SPsigmapc = st.session_state['SPsigmapc_val'] # We won't use this one other than for the value in the number_input and slider widgets 
SPsigma = SPsigmapc/100

## Variables used in heatmap tab.
HMS = st.session_state['HMS_val']
HMK = st.session_state['HMK_val']
HMrpc = st.session_state['HMrpc_val'] # We won't use this one other than for the value in the number_input and slider widgets for interest.
HMr = HMrpc/100
HMtau_years = st.session_state['HMtau_years_val'] 
HMtau_months = st.session_state['HMtau_months_val']
HMtau_weeks = st.session_state['HMtau_weeks_val']
HMtau_days = st.session_state['HMtau_days_val']
HMtau = HMtau_years + HMtau_months/12 + HMtau_weeks/52 + HMtau_days/365 # This is the value we plug into the call option price formula
HMsigmapc = st.session_state['HMsigmapc_val']
HMsigma = HMsigmapc/100
HMp = call(HMS, HMK, HMr, HMtau, HMsigma)

## Here we define out four tabs.

tab1, tab2, tab3, tab4 = st.tabs(['Option price', 'Implied volatility', 'Spot vs premium', 'Heatmap'])

## Now we get into each tab, one by one.

with tab1: # Straight up call/put option price calculated.
   
    # Heading 
    st.header('Black-Scholes European option price')

    # A word of warning
    st.write('_WAIT for Streamlit to stop running before altering any parameter. Do not modify any parameter(s) in rapid succession._')

    # Display the option prices
    st.text(f'Call option price: {call(S, K, r, tau, sigma):.2f}' + ' | ' + f'Put option price: {put(S, K, r, tau, sigma):.2f}')

    # number_input widgets ("boxes")
    S_box = st.number_input('Spot price', min_value=1.00, max_value=1000.00, step=0.01, key='S_box', on_change = update)
    K_box = st.number_input('Strike price', min_value=1.00, max_value=1000.00, step=0.01, key='K_box', on_change = update)
    rpc_box = st.number_input('Annualized risk-free interest rate as a percentage (e.g. if 1%, enter 1)', min_value=0.00, step=0.25, key='rpc_box', on_change = update)
    st.write('Time to maturity')
    years, months, weeks, days = st.columns(4)
    with years:
        tau_years_box = st.number_input('Years', min_value=0, max_value=10, step=1, key='tau_years_box', on_change = update)
    with months:
        tau_months_box = st.number_input('Months', min_value=0, max_value=12, step=1, key='tau_months_box', on_change = update)
    with weeks:
        tau_weeks_box = st.number_input('Weeks', min_value=0, max_value=4, step=1, key='tau_weeks_box', on_change = update)
    with days:
        tau_days_box = st.number_input('Days', min_value=0, max_value=7, step=1, key='tau_days_box', on_change = update)
        
    sigmapc_box = st.number_input('Volatility as a percentage (e.g. if 150%, enter 150)', min_value=0.01, max_value=1000.00, step=0.01, key='sigmapc_box', on_change = update)

    # slider widgets
    S_slider = st.slider('Spot price', min_value=1.00, max_value=1000.00, step=0.01, key='S_slider', on_change = update)
    K_slider = st.slider('Strike price', min_value=1.00, max_value=1000.00, step=0.01, key='K_slider', on_change = update)
    rpc_slider = st.slider('Annualized risk-free interest rate as a percentage (e.g. if 1%, enter 1)', min_value=0.00, max_value=50.00, step=0.25, key='rpc_slider', on_change = update)
    st.write('Time to maturity')
    years, months, weeks, days = st.columns(4)
    with years:
        tau_years_slider = st.slider('Years', min_value=0, max_value=10, step=1, key='tau_years_slider', on_change = update)
    with months:
        tau_months_slider = st.slider('Months', min_value=0, max_value=12, step=1, key='tau_months_slider', on_change = update)
    with weeks:
        tau_weeks_slider = st.slider('Weeks', min_value=0, max_value=4, step=1, key='tau_weeks_slider', on_change = update)
    with days:
        tau_days_slider = st.slider('Days', min_value=0, max_value=7, step=1, key='tau_days_slider', on_change = update)
    sigmapc_slider = st.slider('Volatility as a percentage (e.g. if 150%, enter 150)', min_value=0.01, max_value=1000.00, step=0.01, key='sigmapc_slider', on_change = update)

with tab2: # Implied volatility with plot.

    # Heading 
    st.header('Black-Scholes European call option price')
    st.subheader('Implied volatility')

    # A word of warning
    st.write('_WAIT for Streamlit to stop running before altering any parameter. Do not modify any parameter(s) in rapid succession._')

    # Display the implied volatility
    IVsol = implied_volatility(IVS,IVK,IVr,IVtau,IVp)
    if type(IVsol) == str:
        st.text(f'{IVsol}')
    else: st.text(f'Implied volatility: {100*IVsol:.2f}%.')


    # number_input widgets ("boxes")
    IVp_box = st.number_input('Call option price', min_value=1.00, max_value=1000.00, step=0.01, key='IVp_box', on_change = IVupdate)
    IVS_box = st.number_input('Spot price', min_value=1.00, max_value=1000.00, step=0.01, key='IVS_box', on_change = IVupdate)
    IVK_box = st.number_input('Strike price', min_value=1.00, max_value=1000.00, step=0.01, key='IVK_box', on_change = IVupdate)
    IVrpc_box = st.number_input('Annualized risk-free interest rate as a percentage (e.g. if 1%, enter 1)', min_value=0.00, max_value=50.00, step=0.25, key='IVrpc_box', on_change = IVupdate)
    
    st.write('Time to maturity')
    years, months, weeks, days = st.columns(4)
    with years:
        IVtau_years_box = st.number_input('Years', min_value=0, max_value=10, step=1, key='IVtau_years_box', on_change = IVupdate)
    with months:
        IVtau_months_box = st.number_input('Months', min_value=0, max_value=12, step=1, key='IVtau_months_box', on_change = IVupdate)
    with weeks:
        IVtau_weeks_box = st.number_input('Weeks', min_value=0, max_value=4, step=1, key='IVtau_weeks_box', on_change = IVupdate)
    with days:
        IVtau_days_box = st.number_input('Days', min_value=0, max_value=7, step=1, key='IVtau_days_box', on_change = IVupdate)


    # PLOT IMPLIED VOLATILITY

    extreme = extreme_call_option_price(IVS, IVK, IVr, IVtau)
    if type(extreme) == str:
        st.text(extreme)
    else:
        IVvolmin, IVpmin, IVvolmax, IVpmax, IVvstep, IVpstep = extreme_call_option_price(IVS, IVK, IVr, IVtau)
        fig, ax = plt.subplots()
        fig.suptitle('Implied volatility\n' + r'$(S, K, r, \tau) = $' + f'({IVS:.2f}, {IVK:.2f}, {100*IVr:.2f}%, {IVtau:.3f} yrs)')

        # Make the data
        IV = np.linspace(IVvolmin,IVvolmax,100)
        P = []
        for vol in IV:
            price = call(IVS,IVK,IVr,IVtau,vol)
            P.append(price)

        # Plot the data
        ax.plot(P, 100*IV)

        # Plot the particular point on the curve corresonding to the user's call option price input
        point = implied_volatility(IVS,IVK,IVr,IVtau,IVp)
        if type(point) != str:
            ax.plot(IVp, 100*point, 'bo')
            ax.annotate(f'({IVp:.2f},{100*point:.2f}%)', (IVp, 100*point), textcoords="offset points", xytext=(10,-5), ha='left')

        # A grid sometimes helps
        ax.grid(True)

        # Label the axes etc.
        ax.set_xlabel('Call option price')
        ax.set_ylabel('Implied volatility (%)')

        st.pyplot(fig)

    
    # slider widgets
    IVp_slider = st.slider('Call option price', min_value=1.00, value=float(IVp), max_value=1000.00, step=0.01, key='IVp_slider', on_change = IVupdate)
    IVS_slider = st.slider('Spot price', min_value=1.00,  value=IVS, max_value=1000.00, step=0.01, key='IVS_slider', on_change = IVupdate)
    IVK_slider = st.slider('Strike price', min_value=1.00,  value=IVK, max_value=1000.00, step=0.01, key='IVK_slider', on_change = IVupdate)
    IVrpc_slider = st.slider('Annualized risk-free interest rate as a percentage (e.g. if 1%, enter 1)', min_value=0.00,  value=IVrpc, max_value=50.00, step=0.25, key='IVrpc_slider', on_change = IVupdate)
    st.write('Time to maturity')
    years, months, weeks, days = st.columns(4)
    with years:
        IVtau_years_slider = st.slider('Years', min_value=0, max_value=10, step=1, key='IVtau_years_slider', on_change = IVupdate)
    with months:
        IVtau_months_slider = st.slider('Months', min_value=0, max_value=12, step=1, key='IVtau_months_slider', on_change = IVupdate)
    with weeks:
        IVtau_weeks_slider = st.slider('Weeks', min_value=0, max_value=4, step=1, key='IVtau_weeks_slider', on_change = IVupdate)
    with days:
        IVtau_days_slider = st.slider('Days', min_value=0, max_value=7, step=1, key='IVtau_days_slider', on_change = IVupdate)

with tab3: # Spot vs premium contour plots.
   
    # Heading 
    st.header('Black-Scholes European call option price')
    st.subheader('Spot vs premium time contours')

    # A word of warning
    st.write('_WAIT for Streamlit to stop running before altering any parameter. Do not modify any parameter(s) in rapid succession._')
    
    # number_input widgets ("boxes")
    SPK_box = st.number_input('Strike price', min_value=1.00, max_value=1000.00, step=0.01, key='SPK_box', on_change=SPupdate)
    SPrpc_box = st.number_input('Annualized risk-free interest rate as a percentage (e.g. if 1%, enter 1)', min_value=0.00, max_value=50.00, step=0.25, key='SPrpc_box', on_change=SPupdate)  
    SPsigmapc_box = st.number_input('Volatility as a percentage (e.g. if 150%, enter 150)', min_value=0.01, max_value=1000.00, step=0.01, key='SPsigmapc_box', on_change=SPupdate)
    st.write('Time to maturity')
    years, months, weeks, days = st.columns(4)
    with years:
        SPtau_years_box = st.number_input('Years', min_value=0, max_value=10, step=1, key='SPtau_years_box', on_change = SPupdate)
    with months:
        SPtau_months_box = st.number_input('Months', min_value=0, max_value=12, step=1, key='SPtau_months_box', on_change = SPupdate)
    with weeks:
        SPtau_weeks_box = st.number_input('Weeks', min_value=0, max_value=4, step=1, key='SPtau_weeks_box', on_change = SPupdate)
    with days:
        SPtau_days_box = st.number_input('Days', min_value=0, max_value=7, step=1, key='SPtau_days_box', on_change = SPupdate)
    st.button('Add contour', key='SPapply_tau_box', on_click=SPupdate_contours)
    st.write('_WAIT for Streamlit to stop running before clicking \'Add contour\'._')

    
    # PLOT SPOT PRICE VS CALL OPTION PRICE FOR VARIOUS TIMES TO MATURITY

    fig, ax = plt.subplots()
    fig.suptitle('Spot vs premium at time' + r'$\tau$' + '\n' + fr'$(K, r, \sigma) = $ ({SPK:.2f}, {100*SPr}%, {100*SPsigma:.2f}%)')

    # Make the data
    spotmin = max(SPK*(1 - 2*SPsigma),0)
    spotmax = SPK*(1 + 2*SPsigma)
    spot_data = np.linspace(spotmin,spotmax,100)
    for tau in st.session_state['SPtau_contours']:
        premium_data = [call(spot,SPK,SPr,tau,SPsigma) for spot in spot_data]
        ax.plot(spot_data, premium_data, label=fr'$\tau =$ {tau:.2f} yrs')

    # A grid sometimes helps
    ax.grid(True)

    # Label the axes etc.
    ax.set_xlabel('Spot price')
    ax.set_ylabel('Call option price')

    ax.legend()

    st.pyplot(fig)
    
    #slider widgets
    SPK_slider = st.slider('Strike price', min_value=1.00, max_value=1000.00, step=0.01, key='SPK_slider', on_change=SPupdate)
    SPrpc_slider = st.slider('Annualized risk-free interest rate as a percentage (e.g. if 1%, enter 1)', min_value=0.00, max_value=50.00, step=0.25, key='SPrpc_slider', on_change=SPupdate)  
    SPsigmapc_slider = st.slider('Volatility as a percentage (e.g. if 150%, enter 150)', min_value=0.01, max_value=1000.00, step=0.01, key='SPsigmapc_slider', on_change=SPupdate)  
    st.write('Time to maturity')
    years, months, weeks, days = st.columns(4)
    with years:
        SPtau_years_slider = st.slider('Years', min_value=0, max_value=10, step=1, key='SPtau_years_slider', on_change = SPupdate)
    with months:
        SPtau_months_slider = st.slider('Months', min_value=0, max_value=12, step=1, key='SPtau_months_slider', on_change = SPupdate)
    with weeks:
        SPtau_weeks_slider = st.slider('Weeks', min_value=0, max_value=4, step=1, key='SPtau_weeks_slider', on_change = SPupdate)
    with days:
        SPtau_days_slider = st.slider('Days', min_value=0, max_value=7, step=1, key='SPtau_days_slider', on_change = SPupdate)
    st.button('Add contour', key='SPapply_tau_slider', on_click=SPupdate_contours) 
    st.write('_First WAIT for page to refresh after updating widgets._')


with tab4: # Heatmap, database, etc.
   
    # Heading 
    st.header('Black-Scholes European call option price')
    
    # A word of warning
    st.write('_WAIT for Streamlit to stop running before clicking any button or changing parameters. Do not modify any parameter(s) in rapid succession._')

    button1, button2, button3 = st.columns(3)
    with button1:
        # A button for updating the database.
        st.button('Update database', key='HMupdate_database', on_click=HMupdate_database)     
    with button2:
        # A button for showing what's in the database (just the history of parameters used).
        st.button('Show/hide history', key='HMdb_show', on_click=HMshow_db_history)
    with button3:        
        # A button for starting again with a clean slate.
        st.button('Clear database', key='HMclear_database', on_click=HMclear_database)

    HMdb_history_spot = st.empty() # Placeholder. Paramater history will go here (every second time user clicks 'Show/hide' button).

    if st.session_state['HMdb_history'] == 'show':
        # Open database connection, and create a cursor
        conn = sqlite3.connect('options_price_db')
        c = conn.cursor()
        # Check to see if 'HMhistory_table' exists. If it does, existence_check will be a nonempty list; otherwise, existence_check will be an empty list.
        existence_check = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='HMhistory_table'").fetchall()
        if existence_check != []:
            rows_to_display = c.execute('SELECT DISTINCT spot, strike, interest, time, years, months, weeks, days, volatility, premium FROM HMhistory_table').fetchall()
        c.close()
        conn.close()
        
    with HMdb_history_spot.container():
        if st.session_state['HMdb_history'] == 'show':
            if existence_check != []:
                st.write('Here are the parameter choices for which heatmap data has been saved to the database, plus corresponding premium. To hide, click \'Show/hide history\'.')
                HMshow_hist_df = pd.DataFrame(rows_to_display, columns=['Spot', 'Strike', 'Interest (%)', 'Total time in years', 'Years', 'Months', 'Weeks', 'Days', 'Volatility (%)', 'Premium'])
                st.dataframe(HMshow_hist_df, use_container_width=True)
            else:
                st.write('Database is empty. Choose initial parameters and click \'Update database\' to create a table.')                
            
    # Display the option price
    # Recall that HMp = call(HMS, HMK, HMr, HMtau, HMsigma)
    st.text(f'Call option price: {HMp:.2f}')    

    # number_input widgets ("boxes")
    HMS_box = st.number_input('Spot price', min_value=1.00, max_value=1000.00, step=0.01, key='HMS_box', on_change = HMupdate)
    HMK_box = st.number_input('Strike price', min_value=1.00, max_value=1000.00, step=0.01, key='HMK_box', on_change = HMupdate)
    HMrpc_box = st.number_input('Annualized risk-free interest rate as a percentage (e.g. if 1%, enter 1)', min_value=0.00, max_value=50.00, step=0.25, key='HMrpc_box', on_change = HMupdate)  
    HMsigmapc_box = st.number_input('Volatility as a percentage (e.g. if 150%, enter 150)', min_value=0.01, max_value=1000.00, step=0.01, key='HMsigmapc_box', on_change = HMupdate)
    st.write('Time to maturity')
    years, months, weeks, days = st.columns(4)
    with years:
        HMtau_years_box = st.number_input('Years', min_value=0, max_value=10, step=1, key='HMtau_years_box', on_change = HMupdate)
    with months:
        HMtau_months_box = st.number_input('Months', min_value=0, max_value=12, step=1, key='HMtau_months_box', on_change = HMupdate)
    with weeks:
        HMtau_weeks_box = st.number_input('Weeks', min_value=0, max_value=4, step=1, key='HMtau_weeks_box', on_change = HMupdate)
    with days:
        HMtau_days_box = st.number_input('Days', min_value=0, max_value=7, step=1, key='HMtau_days_box', on_change = HMupdate)

    HMdisplay = st.multiselect('Display heatmap/dataframe?', ['Heatmap', 'Dataframe'], key='HMdisplay')

    if 'Heatmap' in HMdisplay or 'Dataframe' in HMdisplay:
        ##    STEP 1. Search database for table with data.
        ##    STEP 2. If not found, prepare to display a message indicating that the database needs to be updated before any heatmap or dataframe can be displayed.        
        ##    STEP 3. If found, use the data to construct an array that will underlie the heatmap and/or dataframe. Note that the table will not have the same dimensions as the array.
        ##    STEP 4. If found, create the heatmap (if selected).
        ##    STEP 5. If found, create the dataframe (if selected).
        ##    STEP 6. Hold spots for heatmap and dataframe.
        ##    STEP 7. If heatmap and/or dataframe are selected, display if there is anything to display, else write an 'update database' message.
           
        ##    STEP 1. Search database for table with data.
        ##    Open a connection to our database, and create a cursor as well.        
            conn = sqlite3.connect('options_price_db')
            c = conn.cursor()
        ##    Check to see whether a table exists at all, then select the relevant data from our table (if it exists).
        ##    Our table has 12 columns (if it exists).
        ##    The first nine contain the current parameters (HMS, HMK, HMr, HMtau, HMtau_years, HMtau_months, HMtau_weeks, HMtau_days, HMsigma) we want to use to generate the heatmap etc.
        ##    The last three respectively contain the changes in volatility, the changes in spot price, and the changes in premiums corresponding to these volatility and spot changes, and HMK, HMr, HMtau.
        ##    We only need to extract the last three columns. Ensure they are ordered, first by change in volatility (descending), then by change in spot price (ascending).
            existence_check = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='HMpremium_table'").fetchall()
            if existence_check == []:
                found_rows = []
            else:
                found_rows = c.execute("SELECT delta_volatility, delta_spot, delta_premium FROM HMpremium_table WHERE (spot, strike, interest, years, months, weeks, days, volatility) =(?,?,?,?,?,?,?,?) ORDER BY delta_volatility DESC, delta_spot ASC", (HMS,HMK,HMr,HMtau_years, HMtau_months, HMtau_weeks, HMtau_days,HMsigma)).fetchall()
            c.close()
            conn.close()
        ##    If found_rows has length zero, we have no data (table does not exist or exists but is empty): go to STEP 2. Otherwise, continue to STEP 3.

        ##    STEP 2.
            if len(found_rows) == 0:
                proceed_with_display = 'No' # Will display 'update database' message below (if 'Heatmap' or 'Dataframe' has been selected by user).
        ##    STEP 3. Construct an array. Our found_rows list consists of mn lists of length three.
        ##    Here, m is the number of volatility changes considered, and n is the number of spot price changes considered.
        ##    We want to convert this into an (m x n) matrix, where entry (i,j) is the premium corresponding to the ith volatility (decreasing order) and jth spot (increasing order).
        ##    In other words, we are converting an (mn x 3) table into an (m x n) matrix.
        ##    It is important that, for each of the m volatility changes in the first row of the table, there are exactly n spot changes in the second row, and that these n spots are the same for every volatility change.
        ##    This follows from the way the data was created originally, and stored in our table.
        ##    For example, for a given (S,K,r,tau,sigma), say we consider two volatility changes v0,v1 and three spot changes s0,s1,s2.
        ##    Then we have 2x3 = 6 premium changes in the third column of a table of 6 rows: found_rows = [[v0,s0,p00],[v0,s1,p01],[v0,s2,p02],[v1,s0,p10],[v1,s1,p11],[v1,s2,p12]].
        ##    We convert this into a (2 x 3) matrix [[p00, p01, p02], [p10, p11, p12]].
        ##    We'll assume we don't know m and n a priori (even though we do: it's HMnum_rows and HMnum_cols). 
            else:
                proceed_with_display = 'Yes'
                j = 0 
                nv = found_rows[0][0] # volatility v0 in our example.
                npr = found_rows[0][2] # premium p00 in our example.
                ns = found_rows[0][1] # spot s0 in our example.
                HMrows = [nv] # start off the index in what will be our dataframe: rows = [v0] in our example.
                HMcols = [ns] # start off the columns in what will be our dataframe: columns = [s0] in our example.
                HMpremium_matrix = [[npr]] # start off the matrix: [[p00]] in our example.
                for i in range(1,len(found_rows)):
                    if found_rows[i][0] == nv: # First time around, this condition holds while the volatility change in the ith row of our table is the same as v0 (in our example).
                        HMpremium_matrix [j].append(found_rows[i][2]) # First time around, in our example, our matrix becomes [[p00,p01]]
                        if j == 0:
                            HMcols.append(found_rows[i][1]) # First time around, cols becomes [s0,s1] in our example.
                        i += 1 # First time around, i becomes 2, the condition above still holds because found_rows[2][0] = v0. We end up back here with matrix [[p00,p01,p02]] and cols = [s0,s1,s2]. 
                    else:            
                        j += 1 # Note that the j == 0 condition above will no longer hold: our cols is complete.
                        nv = found_rows[i][0]
                        npr = found_rows[i][2]
                        HMrows.append(nv) # First time here, rows becomes [v0,v1] in our example.
                        HMpremium_matrix.append([npr]) # Then matrix becomes [[p00,p01,p02],[p10]]...

    ##    STEP 4. Create the heatmap, but only if 'Heatmap' is selected.
    if 'Heatmap' in HMdisplay and proceed_with_display == 'Yes':
        # Create a figure and axes for the heatmap plot
        fig, ax = plt.subplots()
        fig.suptitle('Change in call option price' + '\n' + fr'$(S,K, r, \tau, \sigma) = $ ({HMS:.2f}, {HMK:.2f}, {100*HMr}%, {HMtau:.3f}yrs, {100*HMsigma}%)')

        # The heatmap itself
        HeatMap = ax.imshow(HMpremium_matrix, cmap="gist_rainbow_r")

        # Show all ticks and label them with the respective list entries
        ax.set_yticks(np.arange(len(HMrows)), labels=[f'{v:.2f}' for v in HMrows])
        ax.set_xticks(np.arange(len(HMcols)), labels=[f'{S:.2f}' for S in HMcols])
        
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', va='center', rotation_mode="anchor")

        # Label axes
        ax.set_xlabel('Change in spot price')
        ax.set_ylabel('Change in volatility (%)')

        # Color bar
        cbar = ax.figure.colorbar(HeatMap, ax=ax, label='Change in call option price')

    ##    STEP 5. Create the dataframe, but only if 'Dataframe' is selected.
    if 'Dataframe' in HMdisplay and proceed_with_display == 'Yes':
        HMdf = pd.DataFrame(HMpremium_matrix, columns = [f'{S:.2f}' for S in HMcols], index = [f'{v:.2f}' for v in HMrows])

    ##    STEP 6. Hold spots for heatmap and dataframe.
    HMspot = st.empty()
    HM_df_spot = st.empty() 

    ##    STEP 7. If heatmap and/or dataframe are selected, display them, or display 'update database' message if there is nothing to display. 
    if 'Heatmap' in HMdisplay or 'Dataframe' in HMdisplay:
        if proceed_with_display == 'Yes':
            if 'Heatmap' in HMdisplay:
                with HMspot:
                    st.write(fig)
            if 'Dataframe' in HMdisplay:
                with HM_df_spot.container():
                    st.write(r'Entry in row $i$ column $j$ contains the difference ($C$ stands for call option price):  ' + '\n\n' + r'$C(S + \Delta S, K, r, \tau, \sigma + \Delta\sigma) - C(S, K, r, \tau, \sigma)$. '
                             + '\n\nHere,' + fr'$(S,K, r, \tau,\sigma) = $ ({HMS:.2f}, {HMK:.2f}, {100*HMr}%, {HMtau:.3f}yrs, {100*HMsigma}%), $100\Delta\sigma$ is the label for row $i$ (change in volatility displayed as a percentage), and $\Delta S$ is the label for column $j$. Each cell directly corresponds to a cell in the heatmap (if displayed).')
                    def neg_red(cell_value):
                        if cell_value < 0:
                            return 'color: red;'
                    st.write(HMdf.style.applymap(neg_red))
        if proceed_with_display == 'No':
            st.write('Database has no record corresponding to this choice of paramaters: click \'Update database\'.')
        

##    conn = sqlite3.connect('options_price_db')
##    c = conn.cursor()
##    whole_thing = c.execute('SELECT * FROM HMpremium_table').fetchall()
##    c.close()
##    conn.close()
##    df = pd.DataFrame(whole_thing)
##    st.write(df)
    

    

    
        


    

    
