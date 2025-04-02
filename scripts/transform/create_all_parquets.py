import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transform import create_parquet_grades
from transform import create_parquet_school_levels
from transform import create_parquet_hvp
from transform import create_parquet_student_course_activities
from transform import create_parquet_quizzes
from transform import create_parquet_student_courses
from transform import create_parquet_assignments
from transform import create_parquet_rel_course_activity
from transform import create_parquet_students
from transform import create_parquet_forums
from transform import create_parquet_resources

# Create students parquet
create_parquet_students.create_parquet_students()

# Create grades parquet
create_parquet_school_levels.create_parquet_grades()

# Create rel course activities parquet
create_parquet_rel_course_activity.create_parquet_rel_course_activity()

# Create student course parquet
create_parquet_student_courses.create_parquet_student_courses()

# Create student course activities parquet
create_parquet_student_course_activities.create_parquet_student_course_activities()

# Create parquet for forums
create_parquet_forums.generate_all_metrics()

# Create parquet for resources
create_parquet_resources.generate_all_metrics()

# Create parquet for quizzes
create_parquet_quizzes.generate_quiz_metrics()

# Create parquet for assignments
create_parquet_assignments.generate_assignment_metrics()

# Create parquet for hvp
create_parquet_hvp.generate_hvp_metrics()

# Create parquet for grades
create_parquet_grades.generate_all_metrics()
