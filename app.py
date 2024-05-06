import streamlit as st
import pandas as pd
import plotly.express as px
import joypy 
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import requests 
from streamlit_lottie import st_lottie



# Load CSS styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply CSS styles
local_css("style.css")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://lottie.host/c99c230b-42a1-43f8-9186-f517a59e7b0c/HwDlCsqaoC.json") 

@st.cache_data
def load_data():
    # Ensure the data path is correct and accessible
    df = pd.read_csv('/Users/benedettasabatino/Desktop/website1/dataset_uni_variables.csv', delimiter=';')
    return df

df = load_data() 


def create_sankey(df, gender_filter='All'):
    """
    Create and return a Sankey diagram figure based on the given DataFrame and gender filter.

    :param df: DataFrame with columns 'gender', 'occupation', 'time_preference'
    :param gender_filter: Filter the data by this gender category ('All', 'Male', 'Female', 'Other')
    :return: Plotly figure object of the Sankey diagram
    """
    # Data preprocessing
    df['gender'] = df['gender'].str.capitalize()
    df['occupation'] = df['occupation'].str.lower()
    df['time_preference'] = df['time_preference'].str.lower()
    df['occupation'].fillna('unknown', inplace=True)
    df['time_preference'].fillna('unknown', inplace=True)

    # Data filtering based on gender
    df_filtered = df if gender_filter == 'All' else df[df['gender'] == gender_filter]
    df_grouped = df_filtered.groupby(['occupation', 'time_preference', 'gender']).size().reset_index(name='count')

    # Create label lists for nodes
    occupation_list = df_grouped['occupation'].unique().tolist()
    time_preferences_list = df_grouped['time_preference'].unique().tolist()
    gender_list = [gender_filter] if gender_filter != 'All' else df_grouped['gender'].unique().tolist()
    label_list = occupation_list + time_preferences_list + gender_list

    # Map categories to indices
    occupation_indices = [occupation_list.index(occ) for occ in df_grouped['occupation']]
    time_preferences_indices = [len(occupation_list) + time_preferences_list.index(tp) for tp in df_grouped['time_preference']]
    gender_indices = [len(occupation_list) + len(time_preferences_list) + gender_list.index(gender) for gender in df_grouped['gender']]

    # Links between nodes
    source = occupation_indices + time_preferences_indices
    target = time_preferences_indices + gender_indices
    value = df_grouped['count'].tolist()

    # Colors for each category
    colors = {
        'occupation': {'student': '#8b0000', 'employed': '#00008b', 'homemaker': '#b22222', 'retired': '#1e90ff', 'unemployed': '#4169e1', 'job seeker': '#ff4500', 'unknown': '#808080'},
        'time_preference': {'late evening': '#5E2D79', 'early afternoon': '#ffa500', 'early morning': '#ffff00', 'late afternoon': '#008080', 'late morning': '#0000ff', 'night': '#8a2be2', 'early evening': '#ff69b4', 'unknown': '#808080'},
        'gender': {'male': '#8b0000', 'female': '#00008b', 'other': '#808080'}
    }

    # Assign colors to nodes
    node_colors = ([colors['occupation'].get(occ, '#808080') for occ in occupation_list] +
                   [colors['time_preference'].get(tp, '#808080') for tp in time_preferences_list] +
                   [colors['gender'].get(gender.lower(), '#808080') for gender in gender_list])

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=label_list, color=node_colors),
        link=dict(source=source, target=target, value=value)
    )])
    fig.update_layout(title_text="Sankey Diagram: Occupation and Time Preferences by Gender", font_size=10)
    return fig

def create_violin_plot(df):
    """
    Create a violin plot showing the distribution of age by viewing partner.

    :param df: DataFrame with columns 'age' and 'watch_with'
    :return: Plotly figure object of the violin plot
    """
    # Create the violin plot
    fig = px.violin(df, y='watch_with', x='age', title='Distribution of Age by Viewing Partner',
                    points=False)  # Do not show any points for a cleaner look

    # Choosing specific colors from the custom diverging red-yellow-blue color palette
    fill_color = '#4393c3'  # Medium blue for the fill
    line_color = '#67001f'  # Dark blue for the line

    # Update the layout to simplify and enhance aesthetics
    fig.update_traces(fillcolor=fill_color, line=dict(color=line_color, width=2))
    fig.update_layout(
        xaxis_title='Age',
        yaxis_title='Who They Watch With',
        plot_bgcolor='white',  # Clean background color
        paper_bgcolor='white',
        font=dict(family='Arial', size=12, color='#333'),
        title=dict(x=0.5, y=0.95, font=dict(size=18, color='black')),
        margin=dict(l=100, r=100, t=80, b=80),
        showlegend=False  # Hide the legend for a cleaner appearance
    )

    # Refine grid lines for a more subtle appearance
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Return the configured figure
    return fig



def create_subscription_dashboard(df):
    """
    Create a Streamlit app for visualizing subscription types and details based on selected subscription type.

    :param df: DataFrame with columns 'subscription_type' and 'subscription_period'
    """
    # Prepare the data for the pie chart
    subscription_pie = df['subscription_type'].value_counts().reset_index()
    subscription_pie.columns = ['Subscription Type', 'Count']

    # Define a color map for subscription types
    color_discrete_map = {
        'basic': '#6baed6',
        'standard': '#08519c',
        'premium': '#041a4a',
        'basic with ads': '#c6dbef'
    }

    # Create a dropdown menu to select the subscription type
    selected_subscription_type = st.selectbox("Select Subscription Type", subscription_pie['Subscription Type'].unique())

    # Filter the data based on the selected subscription type
    filtered_data = df[df['subscription_type'] == selected_subscription_type]

    # Display the pie chart
    pie_chart = px.pie(subscription_pie, values='Count', names='Subscription Type',
                        title='Subscription Type Distribution',
                        color='Subscription Type', color_discrete_map=color_discrete_map)

    # Render the pie chart
    st.plotly_chart(pie_chart, use_container_width=True)

    # Show details graph based on the filtered data
    if not filtered_data.empty:
        # Prepare the data for the secondary graph
        subscription_period_counts = filtered_data['subscription_period'].value_counts().reset_index()
        subscription_period_counts.columns = ['Subscription Period', 'Count']

        # Retrieve the color from the color_discrete_map
        color = color_discrete_map.get(selected_subscription_type, '#808080')  # Use grey as fallback if the key is not found

        # Display the bar chart for subscription periods
        bar_chart = px.bar(subscription_period_counts, x='Subscription Period', y='Count',
                            title=f'Subscription Period for {selected_subscription_type}',
                            color_discrete_sequence=[color])

        # Render the bar chart
        st.plotly_chart(bar_chart, use_container_width=True)
    else:
        st.write(f"No data available for {selected_subscription_type}")


def create_ridgeline_plot(df):
    """
    Create a ridgeline plot showing the distribution of age by weekly usage time.

    :param df: DataFrame with columns 'weekly_usage_time' and 'age'
    """
    # Define a custom colormap that matches the diverging red-yellow-blue palette
    colors = ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
              '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f']
    custom_colormap = LinearSegmentedColormap.from_list("custom_diverging_rdb", colors, N=256)

    # Generate the ridgeline plot with the custom colormap
    fig, axes = joypy.joyplot(df, by='weekly_usage_time', column='age', colormap=custom_colormap, fade=True, figsize=(10, 8))

    # Adding labels and title
    plt.xlabel('Age')
    plt.ylabel('Weekly Usage Time')
    plt.title('Ridgeline Plot of Age by Weekly Usage Time')

    # Show the plot
    st.pyplot(fig)

def create_donut_chart(df):
    """
    Create a donut chart showing the reasons for not using services.

    :param df: DataFrame with a column 'reason_no_netflix' containing reasons encoded.
    """
    # Count the occurrences of each category
    category_counts = df['reason_no_netflix'].value_counts()

    # Prepare DataFrame for Plotly
    df_reasons = pd.DataFrame({
        'Reasons': category_counts.index,
        'Counts': category_counts.values
    })

    # Define a custom diverging red-yellow-blue color palette
    custom_colors = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7',
                     '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']

    # Create the donut chart with Plotly, using the custom color palette
    fig = px.pie(df_reasons, values='Counts', names='Reasons', title='Reasons for Not Using Services - Donut Chart',
                 color='Reasons', color_discrete_sequence=custom_colors, hole=0.3)

    # Update traces to move labels outside and add a legend
    fig.update_traces(textinfo='percent+label', textposition='outside', hoverinfo='label+percent+value')

    # Adjust the legend to appear on the top right
    fig.update_layout(legend_title_text='Reasons', legend=dict(
        orientation="v",
        yanchor="top",
        y=1.0,
        xanchor="right",
        x=1.25
    ))

    # Show the plot using Streamlit's plotly_chart function
    st.plotly_chart(fig)

def create_radar_chart(df):
    """
    Create a radar chart for user experience feedback based on encoded user ratings.

    :param df: DataFrame containing the columns 'user_friendliness', 'price_quality', and 'satisfaction'
    """
    # Mapping dictionary to convert text ratings to numeric codes
    mapping_dict = {
        'very much': 5,
        'very': 4,
        'enough': 3,
        'a little': 2,
        'not at all': 1
    }

    # Replace the textual descriptions with numeric codes and convert to numeric
    df['user_friendliness_encoded'] = pd.to_numeric(df['user_friendliness'].replace(mapping_dict), errors='coerce')
    df['price_quality_encoded'] = pd.to_numeric(df['price_quality'].replace(mapping_dict), errors='coerce')
    df['satisfaction_encoded'] = pd.to_numeric(df['satisfaction'].replace(mapping_dict), errors='coerce')

    # Define a custom colormap that matches the diverging red-yellow-blue palette
    colors = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7',
              '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']
    custom_colormap = LinearSegmentedColormap.from_list("custom_diverging_rdb", colors, N=256)

    # Data for plotting
    labels = ['User Friendly', 'Price Quality Ratio', 'Satisfaction']
    num_vars = len(labels)
    values = [
        df['user_friendliness_encoded'].mean(),
        df['satisfaction_encoded'].mean(),
        df['price_quality_encoded'].mean(),
    ]
    values += values[:1]  # Repeat the first value to close the circular chart

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Repeat the first angle to close the circle

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels, color='darkred', size=15)
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="darkred", size=12)
    plt.ylim(0, 5)

    plot_color = '#b2182b'  # A shade of red from the palette
    ax.plot(angles, values, linewidth=3, linestyle='solid', label='User Feedback', color=plot_color)
    ax.fill(angles, values, color=plot_color, alpha=0.4)

    ax.set_facecolor('whitesmoke')  # Setting the background color to a light gray
    fig.patch.set_facecolor('white')  # Setting the figure background color to white

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('User Experience Feedback - Radar Chart', size=20, color='darkred', y=1.05)

    # Show the plot using Streamlit's pyplot function
    st.pyplot(fig)

def create_sunburst_chart(df):
    """
    Create a Sunburst chart to visualize viewer preferences based on gender, preference, and genre.

    :param df: DataFrame with columns 'gender', 'preference', 'genre_1'
    """
    # Define custom colors
    custom_colors = ['#8B0000', '#00008B', '#d6604d']  # Dark red, dark blue, and burnt orange

    # Create the Sunburst chart directly with Plotly handling the counting
    fig = px.sunburst(
        df,
        path=['gender', 'preference', 'genre_1'],  # Plotly will count occurrences along this path
        title='Viewer Preferences by Gender, Genre, and Episode Category',
        color_discrete_sequence=custom_colors
    )

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    
    # Show the plot using Streamlit's plotly_chart function
    st.plotly_chart(fig)

def creative_plot():
    power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=5da5abe9-605c-459f-a7e0-7888dbe7c5a5&autoAuth=true&ctid=ba9085d7-2255-4a2c-b1cb-dcb1dab5a842"
    
    st.markdown(f'<iframe width="1140" height="541.25" src="{power_bi_url}" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html=True)


# Page 1: User Behaviour
def introduction():

    st.title("Netflix consumer analysis")
    st.subheader("An overview of behaviour, feedback and preferences of the users.")
    st.write(
        "This analysis was conducted through a survey involving approximately 800 respondents to gauge their opinions about one of the major streaming platforms today. /n The comprehensive dataset includes detailed demographic information such as age, gender, marital status, region, and education level, as well as media consumption patterns including subscription types, usage times, and content preferences. This rich array of variables provides deep insights into user behavior and preferences across different viewer segments."
    )
    with st.container():
            st_lottie(lottie_coding, height=300, key="coding")

def user_behaviour_page():
    st.title("User behaviour")

    st.title("Users habits")
    gender_filter = st.selectbox("Select Gender:", ['All', 'Male', 'Female', 'Other'])
    sankey_fig = create_sankey(df, gender_filter)
    st.plotly_chart(sankey_fig)

    st.title("Subscription behaviour")
    create_subscription_dashboard(df)

    st.title("Watch behaviour")
    fig = create_violin_plot(df)
    st.plotly_chart(fig)

    st.title("Weekly usage trends")
    create_ridgeline_plot(df)

# Page 2: User Feedback
def user_feedback_page():
    st.title("User feedback")

    st.title("Reason for not having netflix")
    create_donut_chart(df)
    
    st.title("Client feedback")
    create_radar_chart(df)

# Page 3: Sunburst Chart
def sunburst_page():
    st.title("User preferences")
    
    st.title("Preferred genres")
    create_sunburst_chart(df)

    st.title("Time preferences in the different platforms")
    creative_plot()

# Setup pages
page = st.sidebar.selectbox("Choose a page", ["Dataset introduction","User Behaviour", "User Feedback", "User Preferences"])

if page == "Dataset introduction":
    introduction()
elif page == "User Behaviour":
    user_behaviour_page()
elif page == "User Feedback":
    user_feedback_page()
elif page == "User Preferences":
    sunburst_page()
