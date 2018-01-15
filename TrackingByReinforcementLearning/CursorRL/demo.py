from CursorRL.Environment import Environment
from CursorRL.ExperiencedStudiesBuffer import ExperiencedStudiesBuffer

if __name__ == '__main__':

    env = Environment("D:/bitbucket/projectHandTrack/BoardGame/workFolder/", "./dummy_stat")
    experienced_studies_buffer = ExperiencedStudiesBuffer(500)
    env.reset()
    s = env.start()

    for i in range(9):
        s1, r, d = env.step(0)
        print("{}, {}".format(r, d))

    s1, r, d = env.step(1)
    print("{}, {}".format(r, d))

    for i in range(8):
        s1, r, d = env.step(0)
        print("{}, {}".format(r, d))

    s1, r, d = env.step(1)
    print("{}, {}".format(r, d))

    # for i in range(33):
    #     s1, r, d = env.step(11)
    #     print("{}, {}".format(r, d))
    #
    # for i in range(5):
    #     s1, r, d = env.step(0)
    #     print("{}, {}".format(r, d))
    #
    # s1, r, d = env.step(11)
    # print("{}, {}".format(r, d))
    # print("##################")
    #
    # experienced_studies_buffer.append(env.current_study)
    #
    # experienced_studies_buffer.select_study()
    # warm_frames = experienced_studies_buffer.get_warm_frames()
    #
    # done_trace = False
    #
    # while not done_trace:
    #     # train_batch contains list of [s, a, r, s1, d]
    #     train_batch, trace_length, done_trace = experienced_studies_buffer.get_trace()
    #     print("len(train_batch): {}".format(len(train_batch)))
    #     print("trace_length: {}".format(trace_length))
    #     print("@@@")