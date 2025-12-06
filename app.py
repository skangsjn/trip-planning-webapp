import io
import os
import sqlite3 as sl
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for, send_file, jsonify
from datetime import datetime
import pytz
import calendar
from JPN_data import final_df
import secrets
from pathlib import Path

pd.set_option('display.max_columns', None)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')

@app.route('/', methods=['GET', 'POST'])
def home():
    est = pytz.timezone('America/New_York')
    tokyo = pytz.timezone('Asia/Tokyo')

    local_time = datetime.now(est).strftime('%A %B %d, %Y %I:%M %p')
    japan_time = datetime.now(tokyo).strftime('%A %B %d, %Y %I:%M %p')
    time_difference_str = "Japan is 13 hours ahead of New York."

    # Values stored in the session from previous valid submissions
    start_date_display = session.get('start_date')
    end_date_display = session.get('end_date')
    days_until_trip = session.get('days_until_trip')

    error_message = None

    if request.method == 'POST':
        start_str = request.form.get('start_date')
        end_str = request.form.get('end_date')

        try:
            # HTML date inputs send YYYY-MM-DD
            start_dt = datetime.strptime(start_str, '%Y-%m-%d')
            end_dt = datetime.strptime(end_str, '%Y-%m-%d')
        except (TypeError, ValueError):
            error_message = "Please select a valid start and end date."
        else:
            today = datetime.now(est).date()

            if end_dt < start_dt:
                error_message = "End date must be on or after the start date."
            elif end_dt.date() < today:
                # Entire range is in the past
                error_message = "Please choose a trip window in the future. Past trips can't be planned here."
            else:
                # Valid future or ongoing trip
                if start_dt.date() <= today <= end_dt.date():
                    # Trip has started or starts today
                    days_until_trip = 0
                else:
                    # Trip is in the future
                    days_until_trip = (start_dt.date() - today).days

                start_date_display = start_dt.strftime('%A %B %d, %Y')
                end_date_display = end_dt.strftime('%A %B %d, %Y')

                session['start_date'] = start_date_display
                session['end_date'] = end_date_display
                session['days_until_trip'] = days_until_trip

    return render_template(
        'home.html',
        local_time=local_time,
        japan_time=japan_time,
        time_difference_str=time_difference_str,
        start_date=start_date_display,
        end_date=end_date_display,
        days_until_trip=days_until_trip,
        error_message=error_message,
    )


def get_calendar_starting_sunday(year, month, start_date=None, end_date=None):
    """
    Build a month matrix starting on Sunday.
    """
    cal = calendar.Calendar(firstweekday=6)  # Week starts on Sunday
    month_days = cal.monthdayscalendar(year, month)

    calendar_list = []
    start = start_date.date() if isinstance(start_date, datetime) else None
    end = end_date.date() if isinstance(end_date, datetime) else None

    for week in month_days:
        week_list = []
        for day in week:
            if day == 0:
                week_list.append({'day': '', 'class': 'empty', 'date': None})
                continue

            current_date = datetime(year, month, day).date()
            class_name = ''

            if start and end:
                if current_date == start:
                    class_name = 'highlight-start'
                elif current_date == end:
                    class_name = 'highlight-end'
                elif start < current_date < end:
                    class_name = 'highlight-range'

            week_list.append(
                {
                    'day': day,
                    'class': class_name,
                    'date': current_date.isoformat(),
                }
            )

        calendar_list.append(week_list)

    return calendar_list


def generate_months_for_trip(start_date=None, end_date=None):
    """
    Return a list of month blocks covering the trip window:
    """
    today = datetime.now()
    if start_date is None:
        start_date = today
    if end_date is None or end_date < start_date:
        end_date = start_date

    current = start_date.replace(day=1)
    last = end_date.replace(day=1)

    months = []
    while current <= last:
        year = current.year
        month = current.month
        weeks = get_calendar_starting_sunday(year, month, start_date, end_date)
        months.append(
            {
                'month_name': calendar.month_name[month],
                'year': year,
                'weeks': weeks,
            }
        )

        if month == 12:
            current = current.replace(year=year + 1, month=1)
        else:
            current = current.replace(month=month + 1)

    return months


def get_city_cluster_options():
    """
    Build drag-and-droppable blocks like "Tokyo – Cluster 3"
    using the clustering output from JPN_data.final_df.
    """
    summary = (
        final_df.groupby(['city', 'cluster'])
        .size()
        .reset_index(name='count')
        .sort_values(['city', 'cluster'])
    )

    options = []
    for _, row in summary.iterrows():
        city = row['city']
        cluster = int(row['cluster'])
        cluster_id = f"{city}{cluster}"  # e.g. "Tokyo3"

        options.append(
            {
                'id': cluster_id,
                'city': city,
                'cluster': cluster,
                'label': f"{city} – Cluster {cluster}",
                'count': int(row['count']),
            }
        )
    return options


def build_cluster_details():
    """
    Build a mapping: cluster_id -> list of activities in that cluster.
    Includes both 'type' (for the right-hand tag) and 'category' (for section grouping).
    """
    details = {}
    has_type = 'type' in final_df.columns
    has_category = 'category' in final_df.columns

    for _, row in final_df.iterrows():
        city = row['city']
        cluster = int(row['cluster'])
        cluster_id = f"{city}{cluster}"

        item = {
            'name': row['name'],
            'city': city,
            'cluster': cluster,
        }
        if has_type:
            item['type'] = row['type']
        if has_category:
            item['category'] = row['category']

        details.setdefault(cluster_id, []).append(item)

    return details


@app.route('/calendar_page')
def calendar_page():
    # Human-readable strings from the session (set on home page)
    start_date_str = session.get('start_date')
    end_date_str = session.get('end_date')

    start_dt = end_dt = None
    if start_date_str:
        try:
            start_dt = datetime.strptime(start_date_str, '%A %B %d, %Y')
        except ValueError:
            start_dt = None

    if end_date_str:
        try:
            end_dt = datetime.strptime(end_date_str, '%A %B %d, %Y')
        except ValueError:
            end_dt = None

    months = generate_months_for_trip(start_dt, end_dt)
    city_clusters = get_city_cluster_options()
    itinerary = session.get('itinerary', {})
    cluster_details = build_cluster_details()

    return render_template(
        'calendar_page.html',
        months=months,
        city_clusters=city_clusters,
        itinerary=itinerary,
        start_date=start_date_str,
        end_date=end_date_str,
        cluster_details=cluster_details,
    )


@app.route('/assign_cluster', methods=['POST'])
def assign_cluster():
    """
    Save a (date, city_cluster) pairing into the user's session.
    Ensure each cluster can only appear on one date.
    """
    data = request.get_json(silent=True) or {}
    date_str = data.get('date')
    cluster_id = data.get('cluster_id')
    label = data.get('label')

    if not date_str or not cluster_id:
        return jsonify({'status': 'error', 'message': 'Missing date or cluster_id'}), 400

    itinerary = session.get('itinerary', {})

    for day, items in list(itinerary.items()):
        filtered = [item for item in items if item.get('cluster_id') != cluster_id]
        if filtered:
            itinerary[day] = filtered
        else:
            itinerary.pop(day)

    day_assignments = itinerary.get(date_str, [])
    if not any(item.get('cluster_id') == cluster_id for item in day_assignments):
        day_assignments.append({'cluster_id': cluster_id, 'label': label})
        itinerary[date_str] = day_assignments

    session['itinerary'] = itinerary

    return jsonify({'status': 'ok', 'items': itinerary[date_str]})



@app.route('/remove_cluster', methods=['POST'])
def remove_cluster():
    """
    Remove a (date, city_cluster) pairing from the user's session.
    """
    data = request.get_json(silent=True) or {}
    date_str = data.get('date')
    cluster_id = data.get('cluster_id')

    itinerary = session.get('itinerary', {})

    if date_str in itinerary:
        itinerary[date_str] = [
            item for item in itinerary[date_str] if item.get('cluster_id') != cluster_id
        ]
        if not itinerary[date_str]:
            itinerary.pop(date_str)
        session['itinerary'] = itinerary

    return jsonify({'status': 'ok', 'items': itinerary.get(date_str, [])})


@app.route('/page2')
def page2():
    return render_template('page2.html')


# Need access to Google maps API for this, free trial ran out D:
# @app.route('/map')
# def map_page():
#     locations = final_df[['name', 'latitude', 'longitude']].values.tolist()
#     return render_template('map.html', api_key='ENTER_WHEN_READY', locations=locations)
# Alternative but not as clean:
@app.route('/map')
def map_page():
    df = (
        final_df[['name', 'latitude', 'longitude', 'city', 'cluster']]
        .dropna(subset=['latitude', 'longitude'])
        .copy()
    )

    df['cluster'] = df['cluster'].astype(int)

    locations = df[['name', 'latitude', 'longitude', 'city', 'cluster']].values.tolist()

    return render_template('map.html', locations=locations)


if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', '1') == '1')