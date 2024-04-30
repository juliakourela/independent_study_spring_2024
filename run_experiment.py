import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#CSV file containing input data
input_filename = 'final_subnational.csv'

#Countries from which to select 1/4 of their total samples randomly, stratified by population
countries_to_stratify = ['United States', 'India']

#Values to predict
dependent_vars = ['Residential EUI (kWh/m2/year)', 'Non-residential EUI (kWh/m2/year)']

#Colors to use in plot
plot_colors = ['red', 'blue']

# If 'true', conducts experiment to predict value of dependent vars per capita
# If 'false', conducts experiment to predict overall value of dependent vars
per_capita = True


def experiment(dependent_var, data, per_capita=False):
    df_encoded = pd.get_dummies(data, columns=['Climate_Zone_JHU'])
    data = data.dropna(subset=[dependent_var, 'Climate_Zone_JHU', 'hdi_2021', 'le_2021', 'eys_2021', 'mys_2021', 'gnipc_2021', 'gdi_2021', 'hdi_f_2021', 'le_f_2021', 'eys_f_2021', 'mys_f_2021', 'gni_pc_f_2021', 'hdi_m_2021', 'le_m_2021', 'eys_m_2021', 'mys_m_2021', 'gni_pc_m_2021', 'ihdi_2021', 'coef_ineq_2021', 'loss_2021', 'ineq_le_2021', 'ineq_edu_2021', 'ineq_inc_2021', 'gii_2021', 'mmr_2021', 'abr_2021', 'se_f_2021', 'se_m_2021', 'pr_f_2021', 'pr_m_2021', 'lfpr_f_2021', 'lfpr_m_2021', 'phdi_2021', 'diff_hdi_phdi_2021', 'co2_prod_2021', 'mf_2021', 'pop_total_2021', 'Health index', 'Educational index ', 'Income index'])
    data = pd.merge(data, df_encoded)

    y = data[dependent_var]
    if per_capita == True:
        X = data[['Climate_Zone_JHU_1', 'Climate_Zone_JHU_2', 'Climate_Zone_JHU_3', 'Climate_Zone_JHU_4', 'Climate_Zone_JHU_5', 'Climate_Zone_JHU_6', 'Climate_Zone_JHU_7', 'Climate_Zone_JHU_8', 'Climate_Zone_JHU_9', 'Climate_Zone_JHU_10', 'Climate_Zone_JHU_11', 'Climate_Zone_JHU_12', 'hdi_2021', 'gdi_2021', 'hdi_f_2021', 'hdi_m_2021', 'ihdi_2021', 'coef_ineq_2021', 'loss_2021', 'ineq_le_2021', 'ineq_edu_2021', 'ineq_inc_2021', 'gii_2021', 'mmr_2021', 'abr_2021', 'lfpr_f_2021', 'lfpr_m_2021', 'phdi_2021', 'diff_hdi_phdi_2021', 'mf_2021', 'Health index', 'Educational index ', 'Income index']]
    else:
        X = data[['population', 'Climate_Zone_JHU_1', 'Climate_Zone_JHU_2', 'Climate_Zone_JHU_3', 'Climate_Zone_JHU_4', 'Climate_Zone_JHU_5', 'Climate_Zone_JHU_6', 'Climate_Zone_JHU_7', 'Climate_Zone_JHU_8', 'Climate_Zone_JHU_9', 'Climate_Zone_JHU_10', 'Climate_Zone_JHU_11', 'Climate_Zone_JHU_12', 'hdi_2021',  'gdi_2021', 'hdi_f_2021', 'hdi_m_2021', 'ihdi_2021', 'coef_ineq_2021', 'loss_2021', 'ineq_le_2021', 'ineq_edu_2021', 'ineq_inc_2021', 'gii_2021', 'mmr_2021', 'abr_2021', 'lfpr_f_2021', 'lfpr_m_2021', 'phdi_2021', 'diff_hdi_phdi_2021', 'mf_2021', 'Health index', 'Educational index ', 'Income index']]
    
    X_train,X_test,y_train,y_test=train_test_split(
        X,y, 
        train_size = 0.80, 
        random_state = 1)

    if per_capita == False:
        data = data[[dependent_var, 'population', 'Climate_Zone_JHU_1', 'Climate_Zone_JHU_2', 'Climate_Zone_JHU_3', 'Climate_Zone_JHU_4', 'Climate_Zone_JHU_5', 'Climate_Zone_JHU_6', 'Climate_Zone_JHU_7', 'Climate_Zone_JHU_8', 'Climate_Zone_JHU_9', 'Climate_Zone_JHU_10', 'Climate_Zone_JHU_11', 'Climate_Zone_JHU_12', 'hdi_2021', 'eys_2021', 'mys_2021', 'gdi_2021', 'hdi_f_2021', 'eys_f_2021', 'mys_f_2021', 'hdi_m_2021', 'eys_m_2021', 'mys_m_2021', 'ihdi_2021', 'coef_ineq_2021', 'loss_2021', 'ineq_le_2021', 'ineq_edu_2021', 'ineq_inc_2021', 'gii_2021', 'mmr_2021', 'abr_2021', 'lfpr_f_2021', 'lfpr_m_2021', 'phdi_2021', 'diff_hdi_phdi_2021', 'mf_2021', 'pop_total_2021', 'Health index', 'Educational index ', 'Income index']]
    else:
        data = data[[dependent_var, 'Climate_Zone_JHU_1', 'Climate_Zone_JHU_2', 'Climate_Zone_JHU_3', 'Climate_Zone_JHU_4', 'Climate_Zone_JHU_5', 'Climate_Zone_JHU_6', 'Climate_Zone_JHU_7', 'Climate_Zone_JHU_8', 'Climate_Zone_JHU_9', 'Climate_Zone_JHU_10', 'Climate_Zone_JHU_11', 'Climate_Zone_JHU_12', 'hdi_2021', 'eys_2021', 'mys_2021', 'gdi_2021', 'hdi_f_2021', 'eys_f_2021', 'mys_f_2021', 'hdi_m_2021', 'eys_m_2021', 'mys_m_2021', 'ihdi_2021', 'coef_ineq_2021', 'loss_2021', 'ineq_le_2021', 'ineq_edu_2021', 'ineq_inc_2021', 'gii_2021', 'mmr_2021', 'abr_2021', 'lfpr_f_2021', 'lfpr_m_2021', 'phdi_2021', 'diff_hdi_phdi_2021', 'mf_2021', 'pop_total_2021', 'Health index', 'Educational index ', 'Income index']]
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    return X, y, X_train, X_test, y_train, y_test, model, data


def feature_importance(X, y, dependent_var, X_train, X_test, y_train, y_test, model, data):
    # Feature Importance
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importance.nlargest(10)  # Get top 10 most important features
    print("Top 10 Features:")
    print(top_features)

    # Correlation Analysis
    correlation_matrix = data.corr()
    target_correlation = correlation_matrix[dependent_var].abs().sort_values(ascending=False)
    print(f"Correlation with Target Variable - {dependent_var}:")
    print(target_correlation)

    # Univariate Feature Selection 
    selector = SelectKBest(score_func=f_regression, k=10)  # Select top 10 features
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print(f"Selected Features - {dependent_var}:")
    print(selected_features)

    # Visualize top features based on feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title(f"Top 10 Features based on Feature Importance - {dependent_var}")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.show()

    # Visualize correlation with the target variable
    plt.figure(figsize=(10, 6))
    sns.barplot(x=target_correlation.values, y=target_correlation.index)
    plt.title(f"Correlation with Target Variable - {dependent_var}")
    plt.xlabel("Correlation Coefficient (Absolute Value)")
    plt.ylabel("Feature")
    plt.show()

    # Visualize selected features from univariate feature selection
    plt.figure(figsize=(10, 6))
    sns.barplot(x=selected_features, y=selected_features)
    plt.title(f"Top 10 Features selected by Univariate Feature Selection - {dependent_var}")
    plt.xlabel("Feature Score")
    plt.ylabel("Feature")
    plt.show()

    return 


def random_forest_reg(data, dependent_vars, colors, fig, axs, per_capita=False):
    feature_importances = []
    for i, var in enumerate(dependent_vars):
        if per_capita == True:
            data = data.dropna(subset=['population'])
            data = data[data['population'] != 0]
            data[f'{var}_per_capita'] = data[var] / data['population']
            X, y, X_train, X_test, y_train, y_test, random_forest, new_data = experiment(f'{var}_per_capita', data, True)
            feature_importances.append([X, y, f'{var}_per_capita', X_train, X_test, y_train, y_test, random_forest, new_data])
            print(f'random forest regressor model for {var}_per_capita:')
        else:
            X, y, X_train, X_test, y_train, y_test, random_forest, new_data = experiment(var, data, False)
            feature_importances.append(X, y, var, X_train, X_test, y_train, y_test, random_forest, new_data)
            print(f'random forest regressor model for {var}:')
        predictions = random_forest.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print("Train Score:", random_forest.score(X_train, y_train))
        print("Test Score:", random_forest.score(X_test, y_test))
        print("Mean Squared Error:", mse)

        # Take the logarithm of actual and predicted values
        y_test_log = np.log(y_test)
        predictions_log = np.log(predictions)

        # Plot actual vs. predicted values for each random forest model
        if per_capita == True:
            axs[i].scatter(y_test_log, predictions_log, alpha=0.5, label=f"{var} Per Capita: Random Forest (Log-Log Scale)", color=colors[i])
            axs[i].set_title(f"{var} Per Capita: Random Forest Regressor")
        else:
            axs[i].scatter(y_test_log, predictions_log, alpha=0.5, label=f"{var}: Random Forest (Log-Log Scale)", color=colors[i])
            axs[i].set_title(f"{var}: Random Forest Regressor")
        axs[i].set_xlabel("Log Actual")
        axs[i].set_ylabel("Log Predicted")
        axs[i].legend()

    plt.tight_layout()
    plt.show()

    # Display visualizations of feature importances for each experiment
    for i in feature_importances:
        feature_importance(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8])


def stratify_by_country(data, country):
    rows = data[data['country'] == country]
    # Split sorted rows into quartiles by population & randomly select rows from each quartile
    sorted_rows = rows.sort_values(by='population')
    quartile_size = len(sorted_rows) // 4
    quartiles = [sorted_rows[i*quartile_size:(i+1)*quartile_size] for i in range(4)]
    selected_rows = [np.random.choice(quartile.index, size=quartile_size//4, replace=False) for quartile in quartiles]
    selected_indices = np.concatenate(selected_rows)
    # Drop unselected rows from data
    data.drop(index=set(sorted_rows.index) - set(selected_indices), inplace=True)
    return data


if __name__ == "__main__":
    data = pd.read_csv(input_filename)

    for country in countries_to_stratify:
        data = stratify_by_country(data, country)

    # Create subplots
    fig, axs = plt.subplots(1, len(dependent_vars), figsize=(15, 6))

    random_forest_reg(data, dependent_vars, plot_colors, fig, axs, per_capita)

