import pandas as pd
import os
import sys
import unicodedata


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.moodle_path_resolver import MoodlePathResolver
from utils.base_script import BaseScript


class TeachersFeaturedProcessor(BaseScript):
    
    def count_update_events_by_teacher(self, year, teachers_df, teacher_logs_df):
        # Filter logs by year
        year_logs_df = teacher_logs_df[teacher_logs_df['year'] == year]

        if year_logs_df.empty:
            return pd.DataFrame(columns=['id_docente', 'update_events_count'])

        # Define update event types we want to count
        update_events = [
            'course_module_updated',
            'course_updated', 
            'course_section_updated',
            'grade_item_updated',
            'calendar_event_updated'
        ]

        # Filter for update events
        update_logs = year_logs_df[
            year_logs_df['eventname'].str.contains('|'.join(update_events), na=False)
        ]

        if update_logs.empty:
            return pd.DataFrame(columns=['id_docente', 'update_events_count'])

        # Initialize list to store results
        teacher_event_counts = []

        # For each teacher, count their update events
        for _, teacher_row in teachers_df.iterrows():
            teacher_id = teacher_row['id_docente']
            moodle_user_id = teacher_row.get('moodle_user_id')
            edukrea_user_id = teacher_row.get('edukrea_user_id')
            
            total_events = 0
            
            # Count Moodle events (always available) - with explicit year filter
            if pd.notna(moodle_user_id) and moodle_user_id != '':
                try:
                    moodle_user_id_int = int(moodle_user_id)
                    moodle_events = update_logs[
                        (update_logs['userid'] == moodle_user_id_int) & 
                        (update_logs['platform'] == 'moodle') &
                        (update_logs['year'] == year)
                    ]
                    total_events += len(moodle_events)
                except (ValueError, TypeError):
                    # Skip if moodle_user_id cannot be converted to int
                    pass
            
            # Count Edukrea events (only for 2025) - with explicit year filter
            if year == 2025 and pd.notna(edukrea_user_id) and edukrea_user_id != '':
                try:
                    edukrea_user_id_int = int(edukrea_user_id)
                    edukrea_events = update_logs[
                        (update_logs['userid'] == edukrea_user_id_int) & 
                        (update_logs['platform'] == 'edukrea') &
                        (update_logs['year'] == year)
                    ]
                    total_events += len(edukrea_events)
                except (ValueError, TypeError):
                    # Skip if edukrea_user_id cannot be converted to int
                    pass
            
            teacher_event_counts.append({
                'id_docente': teacher_id,
                'update_events_count': total_events
            })

        return pd.DataFrame(teacher_event_counts)

    def count_unique_students_by_teacher(self, year, academic_load_df, enrollments_df):
        # Filter data by year
        year_load_df = academic_load_df[academic_load_df['year'] == year]
        year_enrollments_df = enrollments_df[enrollments_df['year'] == year]

        if year_load_df.empty or year_enrollments_df.empty:
            return pd.DataFrame(columns=['id_docente', 'unique_students_count'])

        # Create a mapping of teacher to their grade-sede combinations
        teacher_grades = year_load_df.groupby('id_docente').agg({
            'id_grado': lambda x: list(x.unique()),
            'sede': lambda x: list(x.unique())
        }).reset_index()

        # Initialize list to store results
        teacher_student_counts = []

        # For each teacher, count unique students in their grades and sede
        for _, teacher_row in teacher_grades.iterrows():
            teacher_id = teacher_row['id_docente']
            teacher_grades_list = teacher_row['id_grado']
            teacher_sedes_list = teacher_row['sede']

            # Find all students in the grades and sedes this teacher teaches
            students_for_teacher = year_enrollments_df[
                (year_enrollments_df['id_grado'].isin(teacher_grades_list)) &
                (year_enrollments_df['sede'].isin(teacher_sedes_list))
            ]

            # Count unique students
            unique_students_count = students_for_teacher['documento_identificación'].nunique()
            
            teacher_student_counts.append({
                'id_docente': teacher_id,
                'unique_students_count': unique_students_count
            })
        return pd.DataFrame(teacher_student_counts)

    def process_teachers_by_year(self, year, teachers_df, academic_load_df, enrollments_df=None, teacher_logs_df=None):
        # Filter academic load by year first
        year_load_df = academic_load_df[academic_load_df['year'] == year]
        
        # If no data for this year, return empty DataFrame
        if year_load_df.empty:
            return pd.DataFrame()

        # Calculate total subjects and total hours for each teacher
        teacher_stats = year_load_df.groupby('id_docente').agg({
            'id_asignatura': 'nunique',  # Count unique subjects
            'intensidad': 'sum'       # Sum total hours
        }).reset_index()

        teacher_stats.columns = ['id_docente', 'total_subjects', 'total_hours']

        # Count unique students
        student_counts = self.count_unique_students_by_teacher(year, academic_load_df, enrollments_df)
        teacher_stats = teacher_stats.merge(student_counts, on='id_docente', how='left')
        teacher_stats['unique_students_count'] = teacher_stats['unique_students_count'].fillna(0).astype(int)

        # Count update events if teacher logs data is provided
        if teacher_logs_df is not None:
            event_counts = self.count_update_events_by_teacher(year, teachers_df, teacher_logs_df)
            teacher_stats = teacher_stats.merge(event_counts, on='id_docente', how='left')
            teacher_stats['update_events_count'] = teacher_stats['update_events_count'].fillna(0).astype(int)

        # Inner join to only include teachers with academic load in this year
        teachers_year_df = teachers_df.merge(teacher_stats, on='id_docente', how='inner')

        # Add year column and calculate experience/age for this specific year
        teachers_year_df['year'] = year
        teachers_year_df['years_experience_ficc'] = year - teachers_year_df['año_inicio_ficc'].fillna(0).astype(int)
        teachers_year_df['years_experience_total'] = year - teachers_year_df['año_inicio_laboral'].fillna(0).astype(int)
        teachers_year_df['age'] = year - teachers_year_df['año_nacimiento'].fillna(0).astype(int)

        # Fill NaN values with 0 for the stats (shouldn't be needed with inner join, but just in case)
        teachers_year_df['total_subjects'] = teachers_year_df['total_subjects'].fillna(0).astype(int)
        teachers_year_df['total_hours'] = teachers_year_df['total_hours'].fillna(0).astype(int)

        return teachers_year_df

    def process_teachers(self):
        teachers_raw_file = "data/raw/tablas_maestras/docentes.csv"
        teachers_raw_df = pd.read_csv(teachers_raw_file, dtype={"id_docente": "Int64"})

        teachers_users_file = 'data/interim/moodle/teachers_users.csv'
        teachers_users_df = pd.read_csv(teachers_users_file, dtype={"id_docente": "Int64"})

        # Clean column names to avoid duplicates and merge properly
        # Keep only the required columns from teachers_users_df to avoid duplicates
        teachers_users_clean = teachers_users_df[['id_docente', 'moodle_user_id', 'edukrea_user_id']].copy()
        
        teachers_df = teachers_raw_df.merge(teachers_users_clean, on="id_docente", how="left")

        academic_load_file = "data/raw/tablas_maestras/carga_horaria.csv"
        academic_load_df = pd.read_csv(academic_load_file, dtype={"id_docente": "Int64"})

        # Load enrollments data
        enrollments_file = "data/interim/estudiantes/enrollments.csv"
        try:
            enrollments_df = pd.read_csv(enrollments_file)
            self.logger.info("Successfully loaded enrollments data")
        except FileNotFoundError:
            self.logger.warning("Enrollments file not found. Student counts will not be calculated.")
            enrollments_df = None

        # Load teacher logs data
        teacher_logs_file = "data/interim/moodle/teacher_logs.csv"
        try:
            teacher_logs_df = pd.read_csv(teacher_logs_file)
            self.logger.info("Successfully loaded teacher logs data")
        except FileNotFoundError:
            self.logger.warning("Teacher logs file not found. Update event counts will not be calculated.")
            teacher_logs_df = None

        # Create a list to store dataframes for each year
        teachers_by_year = []
        
        for year in range(2024, 2026):
            teachers_year_df = self.process_teachers_by_year(year, teachers_df, academic_load_df, enrollments_df, teacher_logs_df)
            if not teachers_year_df.empty:  # Only add if there's data for this year
                teachers_by_year.append(teachers_year_df)

        # Concatenate all years into a single dataframe
        if teachers_by_year:
            teachers_featured_df = pd.concat(teachers_by_year, ignore_index=True)
        else:
            teachers_featured_df = pd.DataFrame()
        
        output_file = "data/interim/moodle/teachers_featured.csv"
        self.save_to_csv(teachers_featured_df, output_file)


if __name__ == "__main__":
    processor = TeachersFeaturedProcessor()
    processor.process_teachers()
    processor.logger.info("Teacher logs processed successfully.")
    processor.close()
