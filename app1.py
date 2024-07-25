import streamlit as st
import pandas as pd
import plotly.express as px
import os
import base64
import numpy as np
from geopy.distance import geodesic
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import folium
import io
from shapely.geometry import Polygon
from matplotlib.colors import to_hex

def main():
    st.image('suraks.jpg', width=200)
    st.markdown('<h1 style="color: purple;">Suraksha Diagnostic</h1>', unsafe_allow_html=True)
    st.subheader("Suraksha new Center probable location for (Kolkata)")

    st.markdown("""
        <style>
        .stApp {
            background-image: url("https://wallpaperaccess.com/full/1586344.jpg");
            background-size: cover;
        }
        </style>
        """, unsafe_allow_html=True)

    @st.cache_data
    def load_ward_data():
        return pd.read_excel('complete_ward_data.xlsx')

    @st.cache_data
    def load_suraksha_data():
        return pd.read_csv('Suraksha_coordinates.csv')

    population_df = load_ward_data()
    suraksha_df = load_suraksha_data()

    data_info = {
        "Census Data 2011": "Loaded from the Census of India website.",
        "Ward Details": "Scraped from Wikipedia.",
        "Suraksha Locations": "Coordinates from Google Maps.",
        "Income Data": "Based on house rent rates, data scraped from Housing.com.",
        "Amenities Data": "Data about nearby amenities scraped using Open Street Map and Google API's."
    }

    choice = st.selectbox("**Select Data Information**", list(data_info.keys()))
    st.info(data_info[choice])

    pdf_path = 'static/hierarchy.pdf'
    if os.path.exists(pdf_path):
        pdf_link = f'<a href="data:application/pdf;base64,{base64.b64encode(open(pdf_path, "rb").read()).decode()}" target="_blank">View Hierarchy PDF</a>'
        st.markdown(pdf_link, unsafe_allow_html=True)
    else:
        st.error(f"{pdf_path} not found. Please ensure the PDF file is in the correct location.")

    st.subheader("➡️ Suraksha existing coordinates for Kolkata location")

    if not suraksha_df.empty:
        fig = px.scatter_mapbox(suraksha_df,
                                lat='Latitude',
                                lon='Longitude',
                                hover_name='Location',
                                color_discrete_sequence=["red"],
                                size_max=15,
                                zoom=10,
                                height=300,
                                mapbox_style="carto-positron")
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

        st.plotly_chart(fig)

    population_df['male_to_female_ratio'] = population_df['total_population_males'] / population_df['total_population_females']
    population_df['children_to_total_ratio'] = population_df['person_aged_0_6'] / population_df['total_population']
    population_df['rent_to_income_ratio'] = population_df['annual_rent'] / population_df['required_annual_income']
    population_df['healthcare_per_capita'] = population_df['Healthcare Service Count'] / population_df['total_population']
    population_df['busiest_place_per_capita'] = population_df['Busiest Place Count'] / population_df['total_population']
    population_df['healthcare_sentiment_score'] = population_df['Healthcare Sentiment Polarity'] * (1 - population_df['Healthcare Sentiment Subjectivity'])
    population_df['busiest_place_sentiment_score'] = population_df['Busiest Place Sentiment Polarity'] * (1 - population_df['Busiest Place Sentiment Subjectivity'])

    st.subheader('➡️ Distribution Analysis of various features')

    feature = st.selectbox("Select Feature for Distribution Plot",
                           ['total_population', 'required_annual_income', 'adjusted_required_annual_income', 'literacy_rate'])

    def create_histogram(dataframe, feature, title):
        fig = px.histogram(dataframe, x=feature, nbins=30, title=title,
                           labels={feature: feature.replace('_', ' ').title()},
                           hover_data={feature: ':.2f'}, marginal='box')
        fig.update_layout(yaxis_title='Frequency')
        return fig

    st.plotly_chart(create_histogram(population_df, feature, f'Distribution of {feature.replace("_", " ").title()}'))

    st.subheader("➡️ Finding the correlation with the target variable 'suraksha_center_within_3km'")
    # Calculate population density (example calculation, assuming area is given or can be derived)
    population_df['population_density'] = population_df['total_population'] / (population_df['number_of_household'] * 0.01)  # Example: population per 100 households
    
    # Calculate total amenities by summing relevant columns
    amenities_columns = [
        'Healthcare Service Count', 
        'Busiest Place Count', 
        # Add other amenity-related columns here if available
    ]
    population_df['total_amenities'] = population_df[amenities_columns].sum(axis=1)
    key_features = [
        'number_of_household', 'total_population', 'person_aged_0_6', 'literate_population',
        'total_worker_population', 'non_working_population', 'literacy_rate', 'price_sq_ft',
        'property_sq_ft', 'annual_rent', 'required_annual_income', 'Healthcare Service Count',
        'Busiest Place Count', 'Avg Healthcare Rating', 'Avg Busiest Place Rating',
        'Total Healthcare Review Count', 'Total Busiest Place Review Count', 'healthcare_sentiment_score',
        'busiest_place_sentiment_score', 'population_density', 'total_amenities'
    ]

    def count_nearby_suraksha_centers(lat, lon, centers_df, radius_km=3):
        count = 0
        for _, center in centers_df.iterrows():
            distance = geodesic((lat, lon), (center['Latitude'], center['Longitude'])).km
            if distance <= radius_km:
                count += 1
        return count

    population_df['suraksha_center_within_3km'] = population_df.apply(
        lambda row: count_nearby_suraksha_centers(row['lat'], row['long'], suraksha_df), axis=1)

    key_features_with_target = key_features + ['suraksha_center_within_3km']
    correlation_matrix_with_target = population_df[key_features_with_target].corr()

    plt.figure(figsize=(20, 15))
    sns.heatmap(correlation_matrix_with_target, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix for Key Features with suraksha_center_within_3km')
    st.pyplot(plt)

    target_variable = 'suraksha_center_within_3km'
    correlations_with_target = correlation_matrix_with_target[target_variable].sort_values(ascending=False)

    slab_1_threshold = 0.1
    slab_2_threshold = 0.2

    slab_1_features = correlations_with_target[correlations_with_target.abs() <= slab_1_threshold].index.tolist()
    slab_2_features = correlations_with_target[(correlations_with_target.abs() > slab_1_threshold) & (correlations_with_target.abs() <= slab_2_threshold)].index.tolist()
    slab_3_features = correlations_with_target[correlations_with_target.abs() > slab_2_threshold].index.tolist()

    def generate_detailed_summary(features, correlation_matrix, target_variable, slab_name):
        summary = [f"\n### Summary for {slab_name} Correlated Features (Total: {len(features)}):\n"]
        for feature in features:
            correlation_value = correlation_matrix.loc[feature, target_variable]
            relationship = "positively" if correlation_value > 0 else "negatively"
            summary.append(f"**Feature:** '{feature}'\n**Correlation:** {correlation_value:.2f} ({relationship} correlated)\n")
        return "\n".join(summary)

    slab_1_summary = generate_detailed_summary(slab_1_features, correlation_matrix_with_target, target_variable, "Moderately")
    slab_2_summary = generate_detailed_summary(slab_2_features, correlation_matrix_with_target, target_variable, "Highly")
    slab_3_summary = generate_detailed_summary(slab_3_features, correlation_matrix_with_target, target_variable, "Very Highly")

    with st.expander('Show Moderately Correlated Features Summary'):
        st.markdown(slab_1_summary)
    with st.expander('Show Highly Correlated Features Summary'):
        st.markdown(slab_2_summary)
    with st.expander('Show Very Highly Correlated Features Summary'):
        st.markdown(slab_3_summary)

    highly_correlated_features = slab_3_features + slab_2_features

    # Find the ward with the highest values for each highly correlated feature
    wards_highest_values = []
    for feature in highly_correlated_features:
        highest_value_ward = population_df.loc[population_df[feature].idxmax()]
        wards_highest_values.append({
            'Ward': highest_value_ward['ward'],
            'Feature': feature.replace('_', ' ').title(),
            'Value': highest_value_ward[feature],
            'Region': highest_value_ward['Region']
        })

    # Identify the ward with the highest total population
    highest_total_population = population_df.loc[population_df['total_population'].idxmax()]
    highest_total_population_data = {
        'Ward': highest_total_population['ward'],
        'Feature': 'Total Population',
        'Value': highest_total_population['total_population'],
        'Region': highest_total_population['Region'],
        'Male Population': highest_total_population['total_population_males'],
        'Female Population': highest_total_population['total_population_females']
    }

    # Identify the ward with the highest male population
    highest_male_population = population_df.loc[population_df['total_population_males'].idxmax()]
    highest_male_population_data = {
        'Ward': highest_male_population['ward'],
        'Feature': 'Male Population',
        'Value': highest_male_population['total_population_males'],
        'Region': highest_male_population['Region'],
        'Total Population': highest_male_population['total_population'],
        'Female Population': highest_male_population['total_population_females']
    }

    # Identify the ward with the highest female population
    highest_female_population = population_df.loc[population_df['total_population_females'].idxmax()]
    highest_female_population_data = {
        'Ward': highest_female_population['ward'],
        'Feature': 'Female Population',
        'Value': highest_female_population['total_population_females'],
        'Region': highest_female_population['Region'],
        'Total Population': highest_female_population['total_population'],
        'Male Population': highest_female_population['total_population_males']
    }

    # Create a dataframe for the highest value wards
    highest_value_wards_df = pd.DataFrame(wards_highest_values)

    # Display the highest value wards
    if st.checkbox('Show Wards with the Highest Values for Each Highly Correlated Feature'):
        st.subheader('Wards with the Highest Values for Each Highly Correlated Feature')
        st.dataframe(highest_value_wards_df)

    # Display the highest total population ward
    if st.checkbox('Show Ward with the Highest Total Population'):
        st.subheader('Ward with the Highest Total Population')
        st.write(pd.DataFrame([highest_total_population_data]))

    # Display the highest male population ward
    if st.checkbox('Show Ward with the Highest Male Population'):
        st.subheader('Ward with the Highest Male Population')
        st.write(pd.DataFrame([highest_male_population_data]))

    # Display the highest female population ward
    if st.checkbox('Show Ward with the Highest Female Population'):
        st.subheader('Ward with the Highest Female Population')
        st.write(pd.DataFrame([highest_female_population_data]))

    st.subheader("➡️ Performing clustering (K-means, Hierarchical) to identify natural groupings in the data. Evaluating different clustering methods and metrics")

    # Select relevant features for clustering
    features = population_df[[
        'total_population', 
        'person_aged_0_6', 'literate_population', 'iletrate_population', 'total_worker_population', 'non_working_population',
        'literacy_rate', 'Healthcare Service Count', 'Busiest Place Count', 'price_sq_ft', 'property_sq_ft', 'required_annual_income', 'adjusted_required_annual_income',
        'Avg Healthcare Rating', 'Avg Busiest Place Rating', 'healthcare_sentiment_score', 'busiest_place_sentiment_score', '% Healthcare On Road Side',
        '% Busiest Place On Road Side', 'Avg Distance Between Healthcare Services', 'Avg Distance Between Busiest Places', 'male_to_female_ratio',
        'children_to_total_ratio', 'rent_to_income_ratio', 'healthcare_per_capita', 'busiest_place_per_capita',
        'healthcare_sentiment_score', 'busiest_place_sentiment_score'
    ]].fillna(0)

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-means Clustering
    silhouette_scores_kmeans = []
    K = range(2, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(features_scaled)
        score = silhouette_score(features_scaled, kmeans.labels_)
        silhouette_scores_kmeans.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(K, silhouette_scores_kmeans, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal Number of Clusters (K-means)')
    st.pyplot(plt)

    optimal_clusters_kmeans = K[silhouette_scores_kmeans.index(max(silhouette_scores_kmeans))]
    kmeans = KMeans(n_clusters=optimal_clusters_kmeans, random_state=42, n_init=10, max_iter=300)
    population_df['kmeans_cluster'] = kmeans.fit_predict(features_scaled)
    silhouette_kmeans = max(silhouette_scores_kmeans)
    st.text(f"Optimal number of clusters for K-means: {optimal_clusters_kmeans}")
    st.text(f"K-means Silhouette Score: {silhouette_kmeans}")

    # Hierarchical Clustering
    silhouette_scores_hierarchical = []
    for k in K:
        hierarchical = AgglomerativeClustering(n_clusters=k)
        hierarchical.fit(features_scaled)
        score = silhouette_score(features_scaled, hierarchical.labels_)
        silhouette_scores_hierarchical.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(K, silhouette_scores_hierarchical, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal Number of Clusters (Hierarchical)')
    st.pyplot(plt)

    optimal_clusters_hierarchical = K[silhouette_scores_hierarchical.index(max(silhouette_scores_hierarchical))]
    hierarchical = AgglomerativeClustering(n_clusters=optimal_clusters_hierarchical)
    population_df['hierarchical_cluster'] = hierarchical.fit_predict(features_scaled)
    silhouette_hierarchical = max(silhouette_scores_hierarchical)
    st.text(f"Optimal number of clusters for Hierarchical Clustering: {optimal_clusters_hierarchical}")
    st.text(f"Hierarchical Clustering Silhouette Score: {silhouette_hierarchical}")

    # Visualization of clusters using Plotly
    def visualize_clusters(df, cluster_column, title):
        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon="long",
            hover_name="ward",
            hover_data={
                "total_population": True,
                "literate_population": True,
                "total_worker_population": True,
                "non_working_population": True,
                "literacy_rate": True,
                "Healthcare Service Count": True,
                "Busiest Place Count": True,
                "required_annual_income": True,
                "adjusted_required_annual_income": True,
            },
            color=cluster_column,
            size="total_population",
            color_continuous_scale=px.colors.cyclical.IceFire,
            size_max=15,
            zoom=10,
            mapbox_style="carto-positron"
        )
        fig.update_layout(
            title=title,
            mapbox=dict(center=dict(lat=22.5744, lon=88.3629), zoom=12)
        )
        st.plotly_chart(fig)

    if st.checkbox('Show K-means Clustering Results'):
        visualize_clusters(population_df, 'kmeans_cluster', 'K-means Clustering of Wards')

    if st.checkbox('Show Hierarchical Clustering Results'):
        visualize_clusters(population_df, 'hierarchical_cluster', 'Hierarchical Clustering of Wards')

    st.subheader("➡️ Finding out top location which is at the distance more than 2km from existing suraksha center by giving weightage for all the data which we extracted")
    weights = {
        'number_of_household': 0.2,
        'total_population': 0.2,
        'person_aged_0_6': 0.05,
        'literate_population': 0.2,
        'iletrate_population': 0.1,
        'total_worker_population': 0.2,
        'non_working_population': 0.1,
        'literacy_rate': 0.1,
        'Healthcare Service Count': 0.2,
        'Busiest Place Count': 0.1,
        'price_sq_ft': 0.05,
        'required_annual_income': 0.3,
        'adjusted_required_annual_income': 0.3,
        'Avg Healthcare Rating': 0.2,
        'Avg Busiest Place Rating': 0.1,
        'healthcare_sentiment_score': 0.2,
        'busiest_place_sentiment_score': 0.05,
        '% Healthcare On Road Side': 0.2,
        '% Busiest Place On Road Side': 0.1,
        'Avg Distance Between Healthcare Services': 0.05,
        'Avg Distance Between Busiest Places': 0.05,
        'male_to_female_ratio': 0.05,
        'children_to_total_ratio': 0.05,
        'rent_to_income_ratio': 0.05,
        'healthcare_per_capita': 0.05,
        'busiest_place_per_capita': 0.05
    }
    
    # Normalize the DataFrame
    scaler = StandardScaler()
    features_to_normalize = list(weights.keys())
    normalized_features = scaler.fit_transform(population_df[features_to_normalize])
    normalized_df = pd.DataFrame(normalized_features, columns=features_to_normalize)
    
    # Calculate the composite score for each ward
    normalized_df['composite_score'] = normalized_df.apply(lambda row: sum(row[feature] * weights[feature] for feature in weights), axis=1)
    
    # Add the composite score to the original DataFrame
    population_df['composite_score'] = normalized_df['composite_score']
    
    # Rank the wards based on the composite score
    ranked_wards = population_df.sort_values(by='composite_score', ascending=False)
    ranked_wards['rank'] = range(1, len(ranked_wards) + 1)
    
    # Display the top candidates for new Suraksha centers
    top_candidates = ranked_wards[['ward', 'lat', 'long', 'rank', 'composite_score', 'total_population', 'Healthcare Service Count', 'required_annual_income', 'adjusted_required_annual_income']]
    
    # Calculate the nearest Suraksha center information for the top candidates
    def find_nearest_center(lat, long, centers_df):
        min_distance = float('inf')
        nearest_center = None
        nearest_center_lat = None
        nearest_center_long = None
        for _, row in centers_df.iterrows():
            center_location = (row['Latitude'], row['Longitude'])
            distance = geodesic((lat, long), center_location).km
            if distance < min_distance:
                min_distance = distance
                nearest_center = row['Location']
                nearest_center_lat = row['Latitude']
                nearest_center_long = row['Longitude']
        return nearest_center, min_distance, nearest_center_lat, nearest_center_long
    
    # Apply the function to get the nearest Suraksha center and its distance for each top candidate
    top_candidates['nearest_center_info'] = top_candidates.apply(lambda row: find_nearest_center(row['lat'], row['long'], suraksha_df), axis=1)
    top_candidates['nearest_center'] = top_candidates['nearest_center_info'].apply(lambda x: x[0])
    top_candidates['distance'] = top_candidates['nearest_center_info'].apply(lambda x: x[1])
    top_candidates['nearest_lat'] = top_candidates['nearest_center_info'].apply(lambda x: x[2])
    top_candidates['nearest_long'] = top_candidates['nearest_center_info'].apply(lambda x: x[3])
    
    # Drop the temporary 'nearest_center_info' column
    top_candidates.drop(columns=['nearest_center_info'], inplace=True)
    
    # Filter top candidates to ensure they are at least 2.5 km apart from existing centers and from each other
    final_candidates = []
    for _, candidate in top_candidates.iterrows():
        if all(geodesic((candidate['lat'], candidate['long']), (final_candidate['lat'], final_candidate['long'])).km >= 2 for final_candidate in final_candidates):
            if candidate['distance'] >= 2:
                final_candidates.append(candidate)
    
    final_candidates_df = pd.DataFrame(final_candidates)
    
    # Normalize composite scores to a suitable range for the size property
    scaler = MinMaxScaler((5, 15))
    final_candidates_df['composite_score_normalized'] = scaler.fit_transform(final_candidates_df[['composite_score']])
    
    # Visualization of the top candidates along with existing Suraksha centers
    
    # Plot top candidates for new Suraksha centers
    fig_top_candidates = px.scatter_mapbox(
        final_candidates_df,
        lat="lat",
        lon="long",
        hover_name="ward",
        hover_data={
            "composite_score": True,
            "total_population": True,
            "Healthcare Service Count": True,
            "required_annual_income": True,
            "adjusted_required_annual_income": True
        },
        color="composite_score",
        size="composite_score_normalized",
        color_continuous_scale=px.colors.cyclical.IceFire,
        size_max=15,
        zoom=10,
        mapbox_style="carto-positron"
    )
    
    # Plot existing Suraksha centers
    fig_suraksha = px.scatter_mapbox(
        suraksha_df,
        lat="Latitude",
        lon="Longitude",
        hover_name="Location",
        color_discrete_sequence=["red"],
        size_max=15,
        zoom=10,
        mapbox_style="carto-positron"
    )
    
    # Combine all plots
    fig_top_candidates.add_trace(fig_suraksha.data[0])
    
    # Add polylines for distances
    lines = []
    for _, row in final_candidates_df.iterrows():
        lines.append({
            'type': 'scattermapbox',
            'lat': [row['lat'], row['nearest_lat']],
            'lon': [row['long'], row['nearest_long']],
            'mode': 'lines',
            'line': {'width': 2, 'color': 'blue'},
            'name': f"Distance: {row['distance']:.2f} km",
        })
    
    # Add lines to the map
    for line in lines:
        fig_top_candidates.add_trace(go.Scattermapbox(
            lat=line['lat'],
            lon=line['lon'],
            mode=line['mode'],
            line=line['line'],
            name=line['name']
        ))
    
    # Show the combined map
    fig_top_candidates.update_layout(
        title="Filtered Top Candidates for New Suraksha Centers with Proximity to Existing Centers",
        mapbox=dict(center=dict(lat=22.5744, lon=88.3629), zoom=12),
        legend=dict(yanchor="top", y=1.05, xanchor="left", x=0.01)
    )
    
    if st.button('Show Map'):
        st.plotly_chart(fig_top_candidates)
    
    top_n = st.selectbox('Select Top N Candidates', [5, 10, 20, 'All'])
    
    if top_n != 'All':
        top_n = int(top_n)
        final_candidates_display = final_candidates_df.head(top_n)
    else:
        final_candidates_display = final_candidates_df
    
    # Display detailed analysis of top candidates
    for _, row in final_candidates_display.iterrows():
        with st.expander(f"Details for Ward {row['ward']} (Rank: {row['rank']})"):
            st.write(f"Composite Score: {row['composite_score']:.2f}")
            st.write(f"Total Population: {row['total_population']}")
            st.write(f"Healthcare Service Count: {row['Healthcare Service Count']}")
            st.write(f"Required Annual Income: {row['required_annual_income']}")
            st.write(f"Adjusted Required Annual Income: {row['adjusted_required_annual_income']}")
            st.write(f"Nearest Suraksha Center: {row['nearest_center']} at {row['distance']:.2f} km")
    
    st.download_button(
        label="Download Data as CSV",
        data=final_candidates_display.to_csv(index=False).encode('utf-8'),
        file_name='final_ward_location.csv',
        mime='text/csv'
    )
    
    st.subheader("➡️ Multiple optimal location in each ward where we can go and check the location for new suraksha center nearby")
    
    st.markdown("**-- From the filtered data of top location , as we already find the best location based on healthcare , Now from that location we are trying to find the location which are beside the road and easily commutable and mostly busiest**")
    
    optimal_locations_df = pd.read_excel('location_final.xlsx')
    merged_df1 = pd.merge(optimal_locations_df, population_df[['ward', 'Region', 'lat', 'long']], on=['ward', 'Region'], how='left')
    merged_df1 = merged_df1.rename(columns={'nearest_center': 'Location'})
    merged_df = pd.merge(merged_df1, suraksha_df[['Location', 'Latitude', 'Longitude']], on='Location', how='left')
    
    map_center = [22.5726, 88.3639]
    m = folium.Map(location=map_center, zoom_start=12)
    
    # Add Suraksha Diagnostic centers from suraksha_df
    for _, row in merged_df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=row['Location'],
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    
    # Define a colormap for different wards
    num_wards = len(merged_df['ward'].unique())
    colormap = plt.cm.tab20
    
    ward_colors = {ward: to_hex(colormap(i / num_wards)) for i, ward in enumerate(merged_df['ward'].unique())}
    
    # Create feature groups for each ward and for top 5 and top 10 locations
    ward_feature_groups = {ward: folium.FeatureGroup(name=f'Ward {ward}') for ward in merged_df['ward'].unique()}
    top_5_feature_group = folium.FeatureGroup(name='Top 5 Locations')
    top_10_feature_group = folium.FeatureGroup(name='Top 10 Locations')

    # Add ward centers, optimal locations, and boundaries
    for _, row in merged_df.iterrows():
        hover_info = f"<b>Ward:</b> {row['ward']}<br>" \
                     f"Population Density: {row['population_density']}<br>" \
                     f"Literacy Rate: {row['literacy_rate']}<br>" \
                     f"Region: {row['Region']}<br>" \
                     f"Required Annual Income: {row['required_annual_income']}<br>" \
                     f"Healthcare Service Count: {row['Healthcare Service Count']}<br>" \
                     f"Avg Healthcare Rating: {row['Avg Healthcare Rating']}<br>" \
                     f"Distance: {row['distance']}<br>" \
                     f"Localities: {row['localities']}<br>" \
                     f"Rank: {row['rank']}"

        # Add the ward center marker to the appropriate feature group
        ward_feature_groups[row['ward']].add_child(folium.Marker(
            location=[row['lat'], row['long']],
            popup=folium.Popup(hover_info, max_width=300),
            icon=folium.Icon(color='blue', icon='info-sign')
        ))

        ward_color = ward_colors[row['ward']]

        optimal_locations = []
        for i in range(1, 6):
            optimal_lat = row[f'optimal_lat_{i}']
            optimal_long = row[f'optimal_long_{i}']
            if pd.notna(optimal_lat) and pd.notna(optimal_long):
                popup_text = f"Optimal Location {i} for Ward {row['ward']}"
                ward_feature_groups[row['ward']].add_child(folium.CircleMarker(
                    location=[optimal_lat, optimal_long],
                    radius=8,
                    color=ward_color,
                    fill=True,
                    fill_color=ward_color,
                    fill_opacity=0.5,
                    popup=folium.Popup(popup_text, max_width=300)
                ))
                optimal_locations.append((optimal_long, optimal_lat))  # Note the order for Polygon

                # Draw a dark polyline from the ward center to the optimal location
                folium.PolyLine(
                    locations=[[row['lat'], row['long']], [optimal_lat, optimal_long]],
                    color='black',
                    weight=2,
                    opacity=0.7
                ).add_to(ward_feature_groups[row['ward']])

                # Add to top 5 or top 10 feature groups based on rank
                if row['rank'] <= 5:
                    top_5_feature_group.add_child(folium.CircleMarker(
                        location=[optimal_lat, optimal_long],
                        radius=8,
                        color='green',
                        fill=True,
                        fill_color='green',
                        fill_opacity=0.5,
                        popup=folium.Popup(f"{popup_text}<br>Ward: {row['ward']}", max_width=300)
                    ))
                if row['rank'] <= 10:
                    top_10_feature_group.add_child(folium.CircleMarker(
                        location=[optimal_lat, optimal_long],
                        radius=8,
                        color='yellow',
                        fill=True,
                        fill_color='yellow',
                        fill_opacity=0.5,
                        popup=folium.Popup(f"{popup_text}<br>Ward: {row['ward']}", max_width=300)
                    ))

        # Add boundary around optimal locations if there are at least 3 points
        if len(optimal_locations) >= 3:
            optimal_polygon = Polygon(optimal_locations).convex_hull
            ward_feature_groups[row['ward']].add_child(folium.Polygon(
                locations=[list(reversed(coord)) for coord in optimal_polygon.exterior.coords],
                color=ward_color,
                fill=True,
                fill_color=ward_color,
                fill_opacity=0.2
            ))

    # Add all feature groups to the map
    for ward, feature_group in ward_feature_groups.items():
        m.add_child(feature_group)

    # Add top 5 and top 10 feature groups to the map
    m.add_child(top_5_feature_group)
    m.add_child(top_10_feature_group)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save the map
    m.save('optimal_location_with_LatLong.html')

    # Display the map in Streamlit
    st.components.v1.html(m._repr_html_(), height=400)

    # Download merged_df as Excel
    def to_excel(df):
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()  # Use close() instead of save()
        processed_data = output.getvalue()
        return processed_data

    st.download_button(label='Download multiple optimal location in Ward', data=to_excel(merged_df), file_name='final_optimal_location_in_wards.xlsx')

    # Provide the HTML map for download
    with open('optimal_location_with_LatLong.html', 'rb') as file:
        st.download_button(
            label='Download Map as HTML',
            data=file,
            file_name='optimal_location_with_LatLong.html',
            mime='text/html'
        )

if __name__ == '__main__':
    main()

        

