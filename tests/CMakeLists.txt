ADD_TEST(NAME unit_test 
        COMMAND sh -c "${CMAKE_CURRENT_SOURCE_DIR}/testAll.sh \
        | echo 'Note that your PR can result in the timeout issue of the unit test: 120 seconds.'")

SET_TESTS_PROPERTIES(unit_test PROPERTIES TIMEOUT 120)
