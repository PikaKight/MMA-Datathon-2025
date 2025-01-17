# %%
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error


import matplotlib.pyplot as plt

# %%
raw_data = pd.read_excel("resources/Datathon_data-2025-Raw.xlsx")

# %%
selected_raw_data = raw_data.drop(columns=['Country Name','Country Code', 'Time', 'Time Code'])


cleaned_data = selected_raw_data.replace("..", np.nan).apply(pd.to_numeric, errors='coerce')

cleaned_data.describe()

cleaned_data.columns = cleaned_data.columns.astype(str)


# %%
# Supply Chain Cost = Transportation cost + Trade cost + Operation cost + Other Factor

transportation = ['Air transport, freight (million ton-km) [IS.AIR.GOOD.MT.K1]',
                  "Air transport, registered carrier departures worldwide [IS.AIR.DPRT]",
                  "Energy use (kg of oil equivalent per capita) [EG.USE.PCAP.KG.OE]"]

trade = ['Cost to export, border compliance (US$) [IC.EXP.CSBC.CD]',
         'Cost to import, border compliance (US$) [IC.IMP.CSBC.CD]',
         'Customs and other import duties (% of tax revenue) [GC.TAX.IMPT.ZS]',
         'Merchandise trade (% of GDP) [TG.VAL.TOTL.GD.ZS]']

operation = ['Broad money (% of GDP) [FM.LBL.BMNY.GD.ZS]',
             'Manufacturing, value added (% of GDP) [NV.IND.MANF.ZS]']

weight = {
    'Transport': 0.4,
    'Trade': 0.3,
    'Ops': 0.2,
    'Other': 0.1
}

# %%
cleaned_data = cleaned_data[cleaned_data.isnull().mean(axis=1) <= 0.5]

cleaned_data = cleaned_data.fillna(cleaned_data.mean())

# %%
mmScaler = MinMaxScaler()

transport_data = mmScaler.fit_transform(cleaned_data[transportation])

cleaned_data['Transportation Cost'] = transport_data.mean(axis=1)

trade_data = mmScaler.fit_transform(cleaned_data[trade])

cleaned_data['Trade Cost'] = trade_data.mean(axis=1)

operation_data = mmScaler.fit_transform(cleaned_data[operation])

cleaned_data['Operation Cost'] = operation_data.mean(axis=1)

# %%

cleaned_data['Supply Chain Cost'] = (
    weight["Transport"] * cleaned_data['Transportation Cost'] +
    weight["Trade"] * cleaned_data['Trade Cost'] +
    weight["Ops"] * cleaned_data['Operation Cost'] + 
    weight["Other"] * cleaned_data["Logistics performance index: Overall (1=low to 5=high) [LP.LPI.OVRL.XQ]"]
)

print(cleaned_data['Supply Chain Cost'])

# %%
new_data = cleaned_data.drop(columns=["Logistics performance index: Overall (1=low to 5=high) [LP.LPI.OVRL.XQ]",
                                      'Transportation Cost', 'Trade Cost', 'Operation Cost', 
                                      'Logistics performance index: Ability to track and trace consignments (1=low to 5=high) [LP.LPI.TRAC.XQ]',
                                      'Logistics performance index: Competence and quality of logistics services (1=low to 5=high) [LP.LPI.LOGS.XQ]',
                                      'Logistics performance index: Ease of arranging competitively priced shipments (1=low to 5=high) [LP.LPI.ITRN.XQ]',
                                      'Logistics performance index: Efficiency of customs clearance process (1=low to 5=high) [LP.LPI.CUST.XQ]',
                                      'Logistics performance index: Frequency with which shipments reach consignee within scheduled or expected time (1=low to 5=high) [LP.LPI.TIME.XQ]',
                                      'Logistics performance index: Quality of trade and transport-related infrastructure (1=low to 5=high) [LP.LPI.INFR.XQ]'])

new_data = new_data.drop(columns=transportation)
new_data = new_data.drop(columns=trade)
new_data = new_data.drop(columns=operation)

# %%
target = "Supply Chain Cost"

correlations = new_data.corr()

target_correlations = correlations[target]

plt.figure(figsize=(5, 200))

heat_map = sns.heatmap(target_correlations.to_frame(),
                        cmap="coolwarm",          # Use a visually appealing colormap
                        cbar=True,
                        annot=True,
                        square=True,
                        fmt='.2f',
                        annot_kws={'size': 7.5},
                        cbar_kws={"shrink": 0.8},
                        linewidths=0.5
                       )
plt.show()

# %%
threshold_corr = abs(target_correlations).mean()

print(threshold_corr)

focused_data = new_data[target_correlations[abs(target_correlations) >= threshold_corr].index]

target_col = focused_data[target]

features = focused_data.drop(columns=[target])


# %%
X = features.values
y = target_col.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ridgePipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge_regression', Ridge(alpha=1))
])

ridgePipeline.fit(X_train, y_train)

ridge_pred = ridgePipeline.predict(X_test)

mse = mean_squared_error(y_test, ridge_pred)
rmse = root_mean_squared_error(y_test, ridge_pred)
r2 = r2_score(y_test, ridge_pred)

print(f"""

Mean Square Error: {mse}
Root Mean Squared Error: {rmse}
R2 Score: {r2}

""")

# %%
coef = ridgePipeline.named_steps['ridge_regression'].coef_
feature_names = features.columns

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coef
})

coef_df['abs_coefficient'] = coef_df['Coefficient'].abs()

coef_df = coef_df.sort_values(by='abs_coefficient', ascending=False)

print(coef_df.head(10))

# %%
coef_df = coef_df.sort_values(by='abs_coefficient', ascending=True)

print(coef_df.head(10))

# %%
comparison = pd.DataFrame({
    'Actual': y_test,
    'Predicted': ridge_pred
})

print(comparison.head())

# %%
plt.figure(figsize=(10, 6))

plt.scatter(y_test, ridge_pred, alpha=0.6, label="Predicated SCC")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Actual SCC")  # Ideal line

plt.legend(loc='upper left')

plt.xlabel('Actual SCC')
plt.ylabel('Predicted SCC')
plt.title('Actual vs Predicted SCC')

plt.show()


