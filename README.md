# StyleGANAPI

This API user FastAP to expose some of the funcionality of a StyleGAN model and save the images on a FireBase db. 

Funcionality:
* Gallery: Generate links to visualice the images stored on FireBase.
* Generator : Generate a image using a seed or the latent vector.
* Projector and Mix : Generate a projection of an image so it can be use by styleGAN. This also allow to mix layer betewn two generated images.

This API is the BackEnd for this site:
https://github.com/seradiazpin/ThumbnailArtGenerator
