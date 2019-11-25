import os
import sys
import math

def main(argv):
    
    xdiff=[]
    ydiff=[]
    pxdiff=[]
    pydiff=[]
    pzdiff=[]

    SubAnswers=[]
    AnswerKey=[]

 
    
    subpath="dannowitz_jlab2_submission_20191112.csv"


    filepath = 'ANSWERS.csv'
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            #print(line.strip().split(","))
            #if cnt==11:
            #    break
            AnswerKey.append(line.strip().split(","))
            line = fp.readline()
            cnt += 1
    
    #print(len(AnswerKey))

    
    with open(subpath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            #if cnt==11:
            #    break
            #print(line.strip().split(","))
            SubAnswers.append(line.strip().split(","))
            line = fp.readline()
            cnt += 1
    
    print(len(AnswerKey))
    print(len(SubAnswers))

    for track_index in range(0,len(AnswerKey)):
        for parm in range(0,5):
            if parm ==0:
                xdiff.append(float(AnswerKey[track_index][parm])-float(SubAnswers[track_index][parm]))
            elif parm==1:
                ydiff.append(float(AnswerKey[track_index][parm])-float(SubAnswers[track_index][parm]))
            elif parm==2:
                pxdiff.append(float(AnswerKey[track_index][parm])-float(SubAnswers[track_index][parm]))
            elif parm==3:
                pydiff.append(float(AnswerKey[track_index][parm])-float(SubAnswers[track_index][parm]))
            elif parm==4:
                pzdiff.append(float(AnswerKey[track_index][parm])-float(SubAnswers[track_index][parm]))
            else:
                print("Error in parameter range")


    xrms_weight=0.03 #0.015
    yrms_weight=0.03 #0.015
    pxrms_weight=0.01
    pyrms_weight=0.01
    pzrms_weight=0.011

    xdiff_sumsq=0.
    for diff in xdiff:
        xdiff_sumsq=xdiff_sumsq+(diff*diff)
    
    print(xdiff_sumsq)
    print(len(AnswerKey))

    xdiff_rms=math.sqrt(xdiff_sumsq/float(len(AnswerKey)))
    
    ydiff_sumsq=0.
    for diff in ydiff:
        ydiff_sumsq=ydiff_sumsq+(diff*diff)
    
    ydiff_rms=math.sqrt(ydiff_sumsq/float(len(AnswerKey)))

    pxdiff_sumsq=0.
    for diff in pxdiff:
        pxdiff_sumsq=pxdiff_sumsq+(diff*diff)
    
    pxdiff_rms=math.sqrt(pxdiff_sumsq/float(len(AnswerKey)))

    pydiff_sumsq=0.
    for diff in pydiff:
        pydiff_sumsq=pydiff_sumsq+(diff*diff)
    
    pydiff_rms=math.sqrt(pydiff_sumsq/float(len(AnswerKey)))

    pzdiff_sumsq=0.
    for diff in pzdiff:
        pzdiff_sumsq=pzdiff_sumsq+(diff*diff)
    
    pzdiff_rms=math.sqrt(pzdiff_sumsq/float(len(AnswerKey)))

    rms_values=[]

    print("x: "+str(xdiff_rms/xrms_weight))
    rms_values.append(xdiff_rms/xrms_weight)
    print("y: "+str(ydiff_rms/yrms_weight))
    rms_values.append(ydiff_rms/yrms_weight)
    print("px: "+str(pxdiff_rms/pxrms_weight))
    rms_values.append(pxdiff_rms/pxrms_weight)
    print("py: "+str(pydiff_rms/pyrms_weight))
    rms_values.append(pydiff_rms/pyrms_weight)
    print("pz: "+str(pzdiff_rms/pzrms_weight))
    rms_values.append(pzdiff_rms/pzrms_weight)

    print("================")
    print(subpath.split("_")[0])
    score=0.
    for value in rms_values:
        score=score+value
    print("SCORE: "+str(score))


if __name__ == "__main__":
    main(sys.argv[1:])
