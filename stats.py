import config
import jinja2
import subprocess

emotion_values = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

def gen_stat(data_records):
  counts = {
    'angry': 0,
    'disgust': 0,
    'fear': 0,
    'happy': 0,
    'sad': 0,
    'surprise': 0,
    'neutral': 0
  }

  for record in data_records:
    max_emotion = best_match_emotion(record)
    counts[max_emotion] += 1

  total = 0
  for count in counts.values():
    total += count

  for emotion, count in counts.items():
    counts[emotion] = int(counts[emotion] * 100 / total)

  return counts


def gen_all_stats(records):
  stats = {}
  for emp, data in records.items():
    stats[emp] = gen_stat(data)
  return stats


def render_template(path, stats):
  templateLoader = jinja2.FileSystemLoader(searchpath="./templates")
  templateEnv = jinja2.Environment(loader=templateLoader)
  TEMPLATE_FILE = "template.html"
  template = templateEnv.get_template(TEMPLATE_FILE)
  return template.render(stats=stats)

def to_html(records):
  stats = gen_all_stats(records)

  with open(config.OUT_FILE, 'w') as f:
    f.write(render_template('templates/template.html', stats=stats))


def to_pdf(records):
  stats = gen_all_stats(records)

  print(stats)

  with open(config.OUT_FILE, 'w') as f:
    for emp, data in stats.items():
      f.write(emp)
      f.write(':')
      f.write(str(data))
      f.write('\n')

  subprocess.call(['python', 'txt2pdf.py', config.OUT_FILE])


def best_match_emotion(record):
  best_emotion = None
  max_val = 0

  print(record)

  for emotion in emotion_values:
    if record[emotion] > max_val:
      max_val = record[emotion]
      best_emotion = emotion

  return best_emotion