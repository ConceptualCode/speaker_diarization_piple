import datetime

def time(timestamp):
    return str(datetime.timedelta(seconds=timestamp))

def save_transcript(segments, filename="transcript.txt"):
    with open(filename, "w") as f:
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write("\n" + segment["speaker"] + ' ' + time(segment["start"]) + '\n')
            f.write(segment["text"][1:] + ' ')
