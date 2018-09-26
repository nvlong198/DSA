package demo.args4j;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;
import java.util.regex.Pattern;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;
import cc.mallet.fst.CRF;
import cc.mallet.fst.CRFTrainerByThreadedLabelLikelihood;
import cc.mallet.fst.CRFWriter;
import cc.mallet.fst.PerClassAccuracyEvaluator;
import cc.mallet.fst.SimpleTagger.SimpleTaggerSentence2FeatureVectorSequence;
import cc.mallet.fst.TokenAccuracyEvaluator;
import cc.mallet.fst.TransducerEvaluator;
import cc.mallet.fst.TransducerTrainer;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.iterator.LineGroupIterator;
import cc.mallet.types.InstanceList;

/**
 * This class is used to train a mallet crf model.
 * 
 * @author zatcsc
 *
 */
public class App {


    public App() {}

    /**
     * 
     * @param trainingFile
     * @param testingFile
     * @param modelName
     * @param modelFile
     * @throws Exception
     */
    public void trainModel(String trainingFile, String testingFile, String modelName,
        String modelFile) throws Exception {
        File trainFile = new File(trainingFile);
        File testFile = new File(testingFile);
        Pipe pipe = new SimpleTaggerSentence2FeatureVectorSequence();
        InstanceList training = new InstanceList(pipe);
        training.addThruPipe(
                new LineGroupIterator(new FileReader(trainFile), Pattern.compile("\\s*"), true));
        InstanceList testing = new InstanceList(pipe);
        testing.addThruPipe(
                new LineGroupIterator(new FileReader(testFile), Pattern.compile("\\s*"), true));
        /**
         * model
         */
        // CRF crf = new CRF(trainingData.getDataAlphabet(),
        // trainingData.getTargetAlphabet());
        CRF crf = new CRF(pipe, null);
        /**
         * construct the finite state machine
         */
        crf.addFullyConnectedStatesForLabels();
        /**
         * initialize model's weights
         */
        crf.setWeightsDimensionAsIn(training, false);
        /**
         * trainer with multi-thread settings
         */
        CRFTrainerByThreadedLabelLikelihood crfTrainer =
                new CRFTrainerByThreadedLabelLikelihood(crf, 4);
        /**
         * train
         */

        crfTrainer.train(training);
        /**
         * Note: labels can also be obtained from the target alphabet
         */
        	 evaluator = new TokenAccuracyEvaluator(
                new InstanceList[] {training, testing}, new String[] {"Training", "Testing"});
        TransducerEvaluator perClassEvaluator = new PerClassAccuracyEvaluator(
                new InstanceList[] {training, testing}, new String[] {"Training", "Testing"});
        /**
         * evaluate
         */
        crfTrainer.addEvaluator(evaluator);
        crfTrainer.addEvaluator(perClassEvaluator);
        CRFWriter crfWriter = new CRFWriter(modelName) {
            @Override
            public boolean precondition(TransducerTrainer tt) {
                // save the trained model after training finishes
                return tt.getIteration() % Integer.MAX_VALUE == 0;
            }
        };
        crfTrainer.addEvaluator(crfWriter);
        evaluator.evaluate(crfTrainer);
        perClassEvaluator.evaluate(crfTrainer);
        /**
         * shutdown trainer
         */
        crfTrainer.shutdown();
        /**
         * save the trained model (if CRFWriter is not used)
         */
        FileOutputStream fos = new FileOutputStream(modelFile);
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(crf);
        oos.close();
    }

    private static final class Args {
        @Option(name = "-train", metaVar = "[path]", required = true,
                usage = "path to training file")
        String train;
        @Option(name = "-test", metaVar = "[path]", required = true, usage = "path to testing file")
        String test;
        @Option(name = "-modelname", metaVar = "[path]", required = true, usage = "model name")
        String modelName;
        @Option(name = "-modelfile", metaVar = "[path]", required = true, usage = "model file")
        String modelFile;
    }

    public static void main(String[] argv) {
        final Args args = new Args();
        CmdLineParser parser =
                new CmdLineParser(args, ParserProperties.defaults().withUsageWidth(100));
        try {
            parser.parseArgument(argv);
        } catch (CmdLineException e) {
            parser.printUsage(System.err);
            System.exit(0);
        }
        try {
            new App().trainModel(args.train, args.test, args.modelName, args.modelFile);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
