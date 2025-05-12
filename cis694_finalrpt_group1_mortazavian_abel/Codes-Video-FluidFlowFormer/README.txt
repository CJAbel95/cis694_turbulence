-- CIS 694 Deep Learning Final Project -- Group 1
--
-- Javad Mortazavian Najafabadi, CSU ID: 2866681
-- Christopher Abel, CSU ID: 2846112
--
------------------------------------------

README.txt
----------
- Slides_Final_project.pptx -- Slides that we showed during our presentation
- Final_report_Group1_Mortazavian_Abel_formatted.pdf -- Final report

./Codes-Video-SuperResolution
	./CIS694Final_Abel_Group1.mp4 -- Video demonstrating training and testing of super-resolution
									network.
	./get_jhtdb_dataset.py -- Python code to download data from the JHTDB, and convert to
								image files (.png).
	./super_res -- Contains Python source code for super-resolution work
		SuperResNetworks.py -- Classs for various super-resolution CNNs, along with PyTorch
								Dataset classes, for the super-resolution work.
		SRCNNTrain.py -- Main method for training of the SRCNN variants tested in this work.
		SRCNNTest.py -- Test trained SRCNN networks; display reconstructed images.
		MBSRCNNTrain.py -- Main method for training the Multi-Branch SRCNN network used in
						these experiments.
		MBSRCNNTest.py -- Test the network trained in MBSRCNNTrain.py.
		PSNRTest.py -- Calculate and report mean PSNR for all 250 images in the testing set.

Additional Notes
-----------------
-- To download the dataset we used database_extractor.py scripts. They can be found on our GitHub page.

-- The FluidFlowFormer.py and FluidFlowFormer-test.py are located under python_codes folder, and 
training_slices and testing_slices are located in the main directory. After running the train scrpit 
the generated fluidflowformer_model.h5 should be manually copied under the python_codes (this 
is done intentionally to always keep the last .h5 for comparison), and after that run the test script. 