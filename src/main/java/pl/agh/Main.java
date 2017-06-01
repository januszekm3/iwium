package pl.agh;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesianLogisticRegression;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.*;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.Ridor;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.*;
import weka.classifiers.trees.lmt.LogisticBase;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

class Solver {
    public void solve() throws Exception {
//        List<String> pfs = load("PFS.txt");
        List<String> os = load("OS.txt");
        List<String> age = load("age.txt");
        List<String> bmi = load("BMI.txt");
        List<String> bones = load("bones.txt");
        List<String> featureG = load("featureG.txt");
//        List<String> featureM = load("featureM.txt");
//        List<String> featureN = load("featureN.txt");
        List<String> featureT = load("featureT.txt");
        List<String> liver = load("liver.txt");
        List<String> lungs = load("lungs.txt");
        List<String> lymphGland = load("lymphGland.txt");
        List<String> sex = load("sex.txt");

        List<List<String>> lists = Arrays.asList(age, bmi, bones, featureG,featureT,liver, lungs, lymphGland, sex, os);
        int numberOfElements = os.size();
        List<List<String>> result = filterData(numberOfElements, lists);

        int numberOfAllElements = result.get(0).size();
        int trainingSetSize = (int) (0.8 * numberOfAllElements);
        int testSetSize = numberOfAllElements - trainingSetSize;
        double osMedian = calculateMedian(os);


        FastVector attributes = createWekaAttributes();
        Instances trainingSet = new Instances("Rel", attributes, trainingSetSize);
        trainingSet.setClassIndex(lists.size()-1);
        addData(trainingSet, result, attributes, 0, trainingSetSize, osMedian);

        /**
         * Wszystkie regulowe i drzewiaste, wiecej atrybutow przeanalizowac
         */

        Instances testSet = new Instances("Test", attributes, testSetSize);
        testSet.setClassIndex(lists.size()-1);
        addData(testSet, result, attributes, trainingSetSize, testSetSize, osMedian);

        Classifier classifier = new RandomTree();
        classifier.buildClassifier(trainingSet);

        Evaluation evaluation = new Evaluation(trainingSet);
        evaluation.evaluateModel(classifier, testSet);

        String strSummary = evaluation.toSummaryString();
        System.out.println(strSummary);

    }

    private void addData(Instances trainingSet, List<List<String>> result, FastVector attributes, int start, int setSize, double finalClassMedian) {
        for(int i=start;i<start+setSize;i++) {
            Instance instance = new Instance(result.size());
            for(int j=0;j<result.size()-1;j++) {
                instance.setValue((Attribute) attributes.elementAt(j), Double.valueOf(result.get(j).get(i)));
            }
            if(Double.compare(finalClassMedian, Double.valueOf(result.get(result.size()-1).get(i))) <= 0) {
                instance.setValue((Attribute) attributes.elementAt(result.size()-1), "below");
            } else {
                instance.setValue((Attribute) attributes.elementAt(result.size()-1), "over");
            }

            trainingSet.add(instance);
        }
    }

    private List<List<String>> filterData(int numberOfElements, List<List<String>> lists) {
        List<List<String>> result = new ArrayList<>(lists.size());
        for(int i=0;i<lists.size();i++) result.add(new ArrayList<>());

        for(int i=0;i<numberOfElements;i++) {
            int counter = 0;
            for(List<String> attribute : lists) {
                if(getVal(attribute, i) == 1) {
                    counter++;
                }
            }
            if(counter == lists.size()) {
                for(int j=0;j<lists.size();j++) {
                    result.get(j).add(lists.get(j).get(i));
                }
            }

        }
        return result;
    }

    private int getVal(List<String> values, int i) {
        return values.get(i).isEmpty() ? 0 : 1;
    }

    private FastVector createWekaAttributes() {
        Attribute bmi = new Attribute("bmi");
        Attribute featureG = new Attribute("featureG");
//        Attribute featureM = new Attribute("featureM");
//        Attribute featureN = new Attribute("featureN");
        Attribute featureT = new Attribute("featureT");
        Attribute bones = new Attribute("bones");
        Attribute sex = new Attribute("sex");
        Attribute lungs = new Attribute("lungs");
        Attribute age = new Attribute("age");
        Attribute lymphGland = new Attribute("lymphGland");
        Attribute liver = new Attribute("liver");


        FastVector fvClassVal = new FastVector(2);
        fvClassVal.addElement("below");
        fvClassVal.addElement("over");

        Attribute classAttribute = new Attribute("classAttribute",fvClassVal);

        FastVector fastVector = new FastVector(10);
        fastVector.addElement(age);
        fastVector.addElement(bmi);
        fastVector.addElement(bones);
        fastVector.addElement(featureG);
//        fastVector.addElement(featureM);
//        fastVector.addElement(featureN);
        fastVector.addElement(featureT);
        fastVector.addElement(liver);
        fastVector.addElement(lungs);
        fastVector.addElement(lymphGland);
        fastVector.addElement(sex);
        fastVector.addElement(classAttribute);
        return fastVector;
    }

    private double calculateMedian(List<String> lines) {
        List<Double> values = new ArrayList<>();
        for(String line : lines) {
            if(!line.isEmpty() && !line.startsWith("#")) {
                double value = Double.valueOf(line);
                if(Double.compare(value, 0.0) > 0) {
                    values.add(value);
                }
            }
        }
        values.sort(Double::compare);
        return values.get(values.size() / 2);
    }

    public List<String> load(String fileName) throws URISyntaxException, IOException {
        Path path = Paths.get(getClass().getClassLoader().getResource(fileName).toURI());
        List<String> lines = Files.readAllLines(path);
        return  new LinkedList<>(lines);
    }
}

public class Main {

    public static void main(String[] args) throws Exception {
        new Solver().solve();
    }


}
