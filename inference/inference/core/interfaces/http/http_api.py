async def clip_compare(
                    inference_request: ClipCompareRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Computes similarity scores using the OpenAI CLIP model.

                    Args:
                        inference_request (ClipCompareRequest): The request containing the data to be compared.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        ClipCompareResponse: The response containing the similarity scores.
                    """
                    logger.debug(f"Reached /clip/compare")
                    clip_model_id = load_clip_model(inference_request, api_key=api_key)
                    response = await self.model_manager.infer_from_request(
                        clip_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(clip_model_id, actor, n=2)
                    return response

async def infer_from_workflow(
                workflow_request: WorkflowSpecificationInferenceRequest,
                background_tasks: BackgroundTasks,
            ) -> WorkflowInferenceResponse:
                return await process_workflow_inference_request(
                    workflow_request=workflow_request,
                    workflow_specification=workflow_request.specification,
                    background_tasks=background_tasks if not LAMBDA else None,
                )

