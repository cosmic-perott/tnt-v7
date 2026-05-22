public class Main {

    public static void main(String[] args) {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat image = Imgcodecs.imread("foggy.jpg");

        List<Pipeline> pipelines = List.of(
            new CLAHEPipeline(),
            new DCPPipeline(),
            new SharpenPipeline(),
            new StructuralPipeline()
        );

        for (Pipeline pipeline : pipelines) {

            Mat output = pipeline.process(image);

            String filename =
                "output/" + pipeline.getName() + ".png";

            Imgcodecs.imwrite(filename, output);

            System.out.println("Saved: " + filename);
        }
    }
}
